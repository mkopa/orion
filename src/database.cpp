#include "orion/database.h"
#include <iostream>
#include <map>
#include <set>
#include <mutex>
#include <shared_mutex>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cstdio>
#include <limits>
#include <cstring>
#include <cerrno>
#include <unistd.h>
#include <fcntl.h>
#include <bit>

#ifdef _WIN32
#include <windows.h>
#endif

#include "hnswlib/hnswlib.h"

namespace orion
{
  namespace endian_helpers
  {
    template <typename T>
    void write_le(std::ostream &os, const T &value)
    {
      if constexpr (std::endian::native == std::endian::little)
      {
        os.write(reinterpret_cast<const char *>(&value), sizeof(T));
      }
      else
      {
        T le_value = value;
        char *bytes = reinterpret_cast<char *>(&le_value);
        std::reverse(bytes, bytes + sizeof(T));
        os.write(bytes, sizeof(T));
      }
    }

    template <typename T>
    void read_le(std::istream &is, T &value)
    {
      is.read(reinterpret_cast<char *>(&value), sizeof(T));
      if constexpr (std::endian::native != std::endian::little)
      {
        char *bytes = reinterpret_cast<char *>(&value);
        std::reverse(bytes, bytes + sizeof(T));
      }
    }
  } // namespace endian_helpers

  using namespace endian_helpers;

  template <typename T>
  void write_binary(std::ostream &os, const T &value) { write_le(os, value); }
  template <typename T>
  void read_binary(std::istream &is, T &value) { read_le(is, value); }

  void write_string(std::ostream &os, const std::string &str)
  {
    uint64_t len = str.size();
    write_le(os, len);
    if (len > 0)
      os.write(str.c_str(), static_cast<std::streamsize>(len));
  }

  std::string read_string(std::istream &is)
  {
    uint64_t len;
    read_le(is, len);
    if (len == 0)
      return "";
    std::string str(static_cast<size_t>(len), '\0');
    is.read(&str[0], static_cast<std::streamsize>(len));
    if (static_cast<uint64_t>(is.gcount()) != len)
      throw std::runtime_error("Failed to read string from stream.");
    return str;
  }

  void write_config(std::ostream &os, const Config &cfg)
  {
    write_le(os, cfg.vector_dim);
    write_le(os, cfg.max_elements);
  }

  void read_config(std::istream &is, Config &cfg)
  {
    read_le(is, cfg.vector_dim);
    read_le(is, cfg.max_elements);
  }

  void write_metadata_value(std::ostream &os, const MetadataValue &val);
  MetadataValue read_metadata_value(std::istream &is);
  void write_metadata_value(std::ostream &os, const MetadataValue &val)
  {
    std::visit([&os](auto &&arg)
               {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, int64_t>) {
          uint8_t tag = 0;
          write_binary(os, tag);
          write_le(os, arg);
        } else if constexpr (std::is_same_v<T, double>) {
          uint8_t tag = 1;
          write_binary(os, tag);
          write_le(os, arg);
        } else if constexpr (std::is_same_v<T, std::string>) {
          uint8_t tag = 2;
          write_binary(os, tag);
          write_string(os, arg);
        } },
               val);
  }
  MetadataValue read_metadata_value(std::istream &is)
  {
    uint8_t type_index;
    read_binary(is, type_index);
    switch (type_index)
    {
    case 0:
    {
      int64_t v;
      read_le(is, v);
      return v;
    }
    case 1:
    {
      double v;
      read_le(is, v);
      return v;
    }
    case 2:
    {
      return read_string(is);
    }
    default:
      throw std::runtime_error("Invalid variant type tag during deserialization.");
    }
  }

  class Database::Impl
  {
  public:
    struct VectorData
    {
      Vector vector;
      Metadata metadata;
    };
    using InvertedIndex = std::map<std::string, std::map<MetadataValue, std::set<VectorId>>>;

    std::string db_path;
    Config config;
    std::map<VectorId, VectorData> storage;
    InvertedIndex metadata_index;
    hnswlib::L2Space space;
    hnswlib::HierarchicalNSW<float> *hnsw_index = nullptr;
    mutable std::shared_mutex rw_mutex;

    Impl(const std::string &path, const Config &cfg) : db_path(path), config(cfg), space(cfg.vector_dim)
    {
      hnsw_index = new hnswlib::HierarchicalNSW<float>(&space, static_cast<size_t>(config.max_elements), 16, 200, true);
    }
    ~Impl() { delete hnsw_index; }

    void remove_from_metadata_index(VectorId id)
    {
      if (storage.find(id) == storage.end())
        return;
      const auto &meta_to_remove = storage.at(id).metadata;
      for (const auto &[key, value] : meta_to_remove)
      {
        auto key_it = metadata_index.find(key);
        if (key_it != metadata_index.end())
        {
          auto val_it = key_it->second.find(value);
          if (val_it != key_it->second.end())
          {
            val_it->second.erase(id);
            if (val_it->second.empty())
              key_it->second.erase(val_it);
          }
          if (key_it->second.empty())
            metadata_index.erase(key_it);
        }
      }
    }

    bool rebuild_index(size_t new_max_elements)
    {
      hnswlib::HierarchicalNSW<float> *new_index = nullptr;
      try
      {
        new_index = new hnswlib::HierarchicalNSW<float>(&space, new_max_elements, 16, 200, true);
        for (const auto &kv : storage)
        {
          new_index->addPoint(kv.second.vector.data(), kv.first);
        }
      }
      catch (const std::exception &e)
      {
        delete new_index;
        std::cerr << "Rebuild failed: " << e.what() << std::endl;
        return false;
      }
      delete hnsw_index;
      hnsw_index = new_index;
      config.max_elements = new_max_elements;
      return true;
    }

    bool save()
    {
      std::lock_guard<std::shared_mutex> lock(rw_mutex);

      const std::string tmp_db_path = db_path + ".tmp";
      const std::string tmp_hnsw_path = db_path + ".hnsw.tmp";

      try
      {
        hnsw_index->saveIndex(tmp_hnsw_path);
      }
      catch (const std::exception &e)
      {
        std::cerr << "Failed to save HNSW index to temporary file: " << e.what() << std::endl;
        return false;
      }

      std::stringstream hnsw_stream;
      uint64_t hnsw_size = 0;
      {
        std::ifstream hnsw_ifs(tmp_hnsw_path, std::ios::binary);
        if (!hnsw_ifs)
        {
          std::remove(tmp_hnsw_path.c_str());
          return false;
        }
        hnsw_stream << hnsw_ifs.rdbuf();
        hnsw_ifs.seekg(0, std::ios::end);
        hnsw_size = hnsw_ifs.tellg();
      }
      std::remove(tmp_hnsw_path.c_str());

      std::ofstream ofs(tmp_db_path, std::ios::binary | std::ios::out | std::ios::trunc);
      if (!ofs)
        return false;

      ofs.write("ORIONDB2", 8);
      uint32_t format_version = 2;
      write_le(ofs, format_version);

      write_config(ofs, config);

      uint64_t storage_count = storage.size();
      write_le(ofs, storage_count);
      for (const auto &kv : storage)
      {
        write_le(ofs, kv.first);
        const Vector &v = kv.second.vector;
        uint64_t vec_len = v.size();
        write_le(ofs, vec_len);
        if (vec_len > 0)
          ofs.write(reinterpret_cast<const char *>(v.data()), static_cast<std::streamsize>(vec_len * sizeof(float)));
        const Metadata &meta = kv.second.metadata;
        uint64_t meta_pairs = meta.size();
        write_le(ofs, meta_pairs);
        for (const auto &m : meta)
        {
          write_string(ofs, m.first);
          write_metadata_value(ofs, m.second);
        }
      }

      std::stringstream meta_idx_stream;
      uint64_t outer_map_size = metadata_index.size();
      write_le(meta_idx_stream, outer_map_size);
      for (const auto &outer : metadata_index)
      {
        write_string(meta_idx_stream, outer.first);
        uint64_t inner_map_size = outer.second.size();
        write_le(meta_idx_stream, inner_map_size);
        for (const auto &inner : outer.second)
        {
          write_metadata_value(meta_idx_stream, inner.first);
          uint64_t id_set_size = inner.second.size();
          write_le(meta_idx_stream, id_set_size);
          for (auto id : inner.second)
            write_le(meta_idx_stream, id);
        }
      }
      meta_idx_stream.seekg(0, std::ios::end);
      uint64_t meta_idx_size = static_cast<uint64_t>(meta_idx_stream.tellg());
      meta_idx_stream.seekg(0, std::ios::beg);

      write_le(ofs, meta_idx_size);
      ofs << meta_idx_stream.rdbuf();

      write_le(ofs, hnsw_size);
      if (hnsw_size > 0)
        ofs << hnsw_stream.rdbuf();

      ofs.flush();

#if defined(_WIN32)
      {
        HANDLE hFile = CreateFileA(tmp_db_path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile != INVALID_HANDLE_VALUE)
        {
          FlushFileBuffers(hFile);
          CloseHandle(hFile);
        }
      }
#else
      int fd_sync = ::open(tmp_db_path.c_str(), O_RDONLY);
      if (fd_sync != -1)
      {
        ::fsync(fd_sync);
        ::close(fd_sync);
      }
#endif

      ofs.close();

      if (std::rename(tmp_db_path.c_str(), db_path.c_str()) != 0)
      {
        std::cerr << "Error: Cannot atomically rename tmp DB file to final DB file: errno=" << errno << std::endl;
        return false;
      }
      return true;
    }

    bool load()
    {
      std::lock_guard<std::shared_mutex> lock(rw_mutex);
      std::ifstream ifs(db_path, std::ios::binary | std::ios::in);
      if (!ifs)
        return false;

      char magic[8];
      ifs.read(magic, 8);
      if (std::memcmp(magic, "ORIONDB2", 8) != 0)
      {
        std::cerr << "Invalid or unsupported DB magic." << std::endl;
        return false;
      }
      uint32_t format_version = 0;
      read_le(ifs, format_version);
      read_config(ifs, config);

      if (hnsw_index)
        delete hnsw_index;
      space = hnswlib::L2Space(config.vector_dim);
      hnsw_index = new hnswlib::HierarchicalNSW<float>(&space, static_cast<size_t>(config.max_elements), 16, 200, true);

      uint64_t storage_count = 0;
      read_le(ifs, storage_count);
      storage.clear();
      for (uint64_t i = 0; i < storage_count; ++i)
      {
        VectorId id;
        read_le(ifs, id);
        uint64_t vec_len = 0;
        read_le(ifs, vec_len);
        Vector v(static_cast<size_t>(vec_len));
        if (vec_len > 0)
          ifs.read(reinterpret_cast<char *>(v.data()), static_cast<std::streamsize>(vec_len * sizeof(float)));
        uint64_t meta_pairs = 0;
        read_le(ifs, meta_pairs);
        Metadata meta;
        for (uint64_t m = 0; m < meta_pairs; ++m)
        {
          std::string key = read_string(ifs);
          MetadataValue mv = read_metadata_value(ifs);
          meta.emplace(std::move(key), std::move(mv));
        }
        storage[id] = {v, meta};
      }

      metadata_index.clear();
      uint64_t meta_idx_size = 0;
      read_le(ifs, meta_idx_size);
      if (meta_idx_size > 0)
      {
        std::string meta_idx_buffer(static_cast<size_t>(meta_idx_size), '\0');
        ifs.read(&meta_idx_buffer[0], static_cast<std::streamsize>(meta_idx_size));
        std::stringstream meta_idx_stream(meta_idx_buffer, std::ios::binary | std::ios::in);

        uint64_t outer_map_size = 0;
        read_le(meta_idx_stream, outer_map_size);
        for (uint64_t i = 0; i < outer_map_size; ++i)
        {
          std::string outer_key = read_string(meta_idx_stream);
          uint64_t inner_map_size = 0;
          read_le(meta_idx_stream, inner_map_size);
          for (uint64_t j = 0; j < inner_map_size; ++j)
          {
            MetadataValue mv = read_metadata_value(meta_idx_stream);
            uint64_t id_set_size = 0;
            read_le(meta_idx_stream, id_set_size);
            std::set<VectorId> ids;
            for (uint64_t k = 0; k < id_set_size; ++k)
            {
              VectorId id_val;
              read_le(meta_idx_stream, id_val);
              ids.insert(id_val);
            }
            metadata_index[outer_key][mv] = std::move(ids);
          }
        }
      }

      uint64_t hnsw_size = 0;
      read_le(ifs, hnsw_size);
      if (hnsw_size > 0)
      {
        std::string hnsw_buffer(static_cast<size_t>(hnsw_size), '\0');
        ifs.read(&hnsw_buffer[0], static_cast<std::streamsize>(hnsw_size));

        const std::string tmp_hnsw_path = db_path + ".hnsw.load.tmp";
        {
          std::ofstream hnsw_ofs(tmp_hnsw_path, std::ios::binary);
          hnsw_ofs.write(hnsw_buffer.data(), hnsw_buffer.size());
        }

        try
        {
          hnsw_index->loadIndex(tmp_hnsw_path, &space);
        }
        catch (const std::exception &e)
        {
          std::cerr << "Warning: failed to load HNSW index from temp file: " << e.what() << std::endl;
        }
        std::remove(tmp_hnsw_path.c_str());
      }

      for (const auto &kv : storage)
      {
        try
        {
          hnsw_index->addPoint(kv.second.vector.data(), kv.first);
        }
        catch (...)
        {
        }
      }
      return true;
    }

    bool add(VectorId id, const Vector &vec, const Metadata &meta)
    {
      if (vec.size() != config.vector_dim)
        return false;
      std::lock_guard<std::shared_mutex> lock(rw_mutex);
      if (storage.count(id))
      {
        remove_from_metadata_index(id);
        try
        {
          hnsw_index->markDelete(id);
        }
        catch (...)
        {
        }
      }
      storage[id] = {vec, meta};
      try
      {
        hnsw_index->addPoint(vec.data(), id);
      }
      catch (const std::exception &e)
      {
        size_t proposed = std::max<size_t>(config.max_elements * 2, storage.size() + 10);
        if (!rebuild_index(proposed))
        {
          return false;
        }
        try
        {
          hnsw_index->addPoint(vec.data(), id);
        }
        catch (const std::exception &e2)
        {
          std::cerr << "Failed to add point after rebuild: " << e2.what() << std::endl;
          return false;
        }
      }
      for (const auto &[key, value] : meta)
      {
        metadata_index[key][value].insert(id);
      }
      return true;
    }

    std::vector<QueryResult> query(const Vector &query_vec, size_t n) const
    {
      std::shared_lock<std::shared_mutex> lock(rw_mutex);
      if (query_vec.size() != config.vector_dim || storage.empty())
        return {};
      auto result_queue = hnsw_index->searchKnn(query_vec.data(), n);
      std::vector<QueryResult> results;
      results.reserve(result_queue.size());
      while (!result_queue.empty())
      {
        results.push_back({result_queue.top().second, result_queue.top().first});
        result_queue.pop();
      }
      std::reverse(results.begin(), results.end());
      return results;
    }

    std::vector<QueryResult> query(const Vector &query_vec, size_t n, const Metadata &filter) const
    {
      if (filter.empty())
        return this->query(query_vec, n);
      std::shared_lock<std::shared_mutex> lock(rw_mutex);
      std::set<VectorId> candidate_ids;
      bool first = true;
      for (const auto &[key, value] : filter)
      {
        auto it_key = metadata_index.find(key);
        if (it_key == metadata_index.end())
          return {};
        auto it_val = it_key->second.find(value);
        if (it_val == it_key->second.end())
          return {};
        const auto &ids_for_clause = it_val->second;
        if (first)
        {
          candidate_ids = ids_for_clause;
          first = false;
        }
        else
        {
          std::set<VectorId> intersection;
          std::set_intersection(candidate_ids.begin(), candidate_ids.end(), ids_for_clause.begin(), ids_for_clause.end(), std::inserter(intersection, intersection.begin()));
          candidate_ids = std::move(intersection);
        }
        if (candidate_ids.empty())
          return {};
      }
      if (candidate_ids.empty())
        return {};
      class IdFilterFunctor : public hnswlib::BaseFilterFunctor
      {
        const std::set<VectorId> &allowed_ids;

      public:
        explicit IdFilterFunctor(const std::set<VectorId> &ids) : allowed_ids(ids) {}
        bool operator()(hnswlib::labeltype id) override { return allowed_ids.count(id); }
      };
      IdFilterFunctor filter_functor(candidate_ids);
      auto result_queue = hnsw_index->searchKnn(query_vec.data(), n, &filter_functor);
      std::vector<QueryResult> results;
      results.reserve(result_queue.size());
      while (!result_queue.empty())
      {
        results.push_back({result_queue.top().second, result_queue.top().first});
        result_queue.pop();
      }
      std::reverse(results.begin(), results.end());
      return results;
    }

    std::optional<std::pair<Vector, Metadata>> get(VectorId id) const
    {
      std::shared_lock<std::shared_mutex> lock(rw_mutex);
      auto it = storage.find(id);
      if (it != storage.end())
        return std::make_pair(it->second.vector, it->second.metadata);
      return std::nullopt;
    }

    bool remove(VectorId id)
    {
      std::lock_guard<std::shared_mutex> lock(rw_mutex);
      if (storage.find(id) == storage.end())
        return false;
      remove_from_metadata_index(id);
      try
      {
        hnsw_index->markDelete(id);
      }
      catch (...)
      {
      }
      storage.erase(id);
      return true;
    }

    size_t count() const
    {
      std::shared_lock<std::shared_mutex> lock(rw_mutex);
      return storage.size();
    }
  };

  Database::Database() : pimpl(nullptr) {}
  Database::~Database() { delete pimpl; }
  Database::Database(Database &&other) noexcept : pimpl(other.pimpl) { other.pimpl = nullptr; }
  Database &Database::operator=(Database &&other) noexcept
  {
    if (this != &other)
    {
      delete pimpl;
      pimpl = other.pimpl;
      other.pimpl = nullptr;
    }
    return *this;
  }
  std::optional<Database> Database::create(const std::string &path, const Config &config)
  {
    try
    {
      Database d;
      d.pimpl = new Impl(path, config);
      if (!d.pimpl->save())
        return std::nullopt;
      return d;
    }
    catch (...)
    {
      return std::nullopt;
    }
  }
  std::optional<Database> Database::load(const std::string &path)
  {
    try
    {
      Database d;
      d.pimpl = new Impl(path, Config());
      if (!d.pimpl->load())
        return std::nullopt;
      return d;
    }
    catch (...)
    {
      return std::nullopt;
    }
  }
  bool Database::save()
  {
    if (!pimpl)
      return false;
    return pimpl->save();
  }
  bool Database::add(VectorId id, const Vector &vec, const Metadata &meta)
  {
    if (!pimpl)
      return false;
    return pimpl->add(id, vec, meta);
  }
  std::vector<QueryResult> Database::query(const Vector &query_vec, size_t n) const
  {
    if (!pimpl)
      return {};
    return pimpl->query(query_vec, n);
  }
  std::vector<QueryResult> Database::query(const Vector &query_vec, size_t n, const Metadata &filter) const
  {
    if (!pimpl)
      return {};
    return pimpl->query(query_vec, n, filter);
  }
  std::optional<std::pair<Vector, Metadata>> Database::get(VectorId id) const
  {
    if (!pimpl)
      return std::nullopt;
    return pimpl->get(id);
  }
  bool Database::remove(VectorId id)
  {
    if (!pimpl)
      return false;
    return pimpl->remove(id);
  }
  size_t Database::count() const
  {
    if (!pimpl)
      return 0;
    return pimpl->count();
  }
  std::string Database::get_version() { return "0.2.0-alpha"; }

} // namespace orion