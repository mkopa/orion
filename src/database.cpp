#include "orion/database.h"
#include <iostream>
#include <map>
#include <set>
#include <mutex>
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

#ifdef _WIN32
#include <windows.h>
#endif

#include "hnswlib/hnswlib.h"

namespace orion
{

  // helpers for binary read/write (host-endian, fixed-size)
  template <typename T>
  void write_binary(std::ostream &os, const T &value) { os.write(reinterpret_cast<const char *>(&value), sizeof(T)); }
  template <typename T>
  void read_binary(std::istream &is, T &value) { is.read(reinterpret_cast<char *>(&value), sizeof(T)); }

  void write_string(std::ostream &os, const std::string &str)
  {
    uint64_t len = str.size();
    write_binary(os, len);
    if (len > 0)
      os.write(str.c_str(), static_cast<std::streamsize>(len));
  }

  std::string read_string(std::istream &is)
  {
    uint64_t len;
    read_binary(is, len);
    if (len == 0)
      return "";
    std::string str(static_cast<size_t>(len), '\0');
    is.read(&str[0], static_cast<std::streamsize>(len));
    if (static_cast<uint64_t>(is.gcount()) != len)
      throw std::runtime_error("Failed to read string from stream.");
    return str;
  }

  // explicit config serialization (safer than raw struct write)
  void write_config(std::ostream &os, const Config &cfg)
  {
    // explicit, version-tolerant serialization
    write_binary(os, cfg.vector_dim);
    write_binary(os, cfg.max_elements);
  }

  void read_config(std::istream &is, Config &cfg)
  {
    read_binary(is, cfg.vector_dim);
    read_binary(is, cfg.max_elements);
  }

  // MetadataValue serialization with explicit type tags (stable)
  void write_metadata_value(std::ostream &os, const MetadataValue &val);
  MetadataValue read_metadata_value(std::istream &is);
  void write_metadata_value(std::ostream &os, const MetadataValue &val)
  {
    // explicit type tagging to avoid dependence on std::variant index ordering
    std::visit([&os](auto &&arg)
               {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, int64_t>) {
          uint8_t tag = 0;
          write_binary(os, tag);
          write_binary(os, arg);
        } else if constexpr (std::is_same_v<T, double>) {
          uint8_t tag = 1;
          write_binary(os, tag);
          write_binary(os, arg);
        } else if constexpr (std::is_same_v<T, std::string>) {
          uint8_t tag = 2;
          write_binary(os, tag);
          write_string(os, arg);
        } else {
          static_assert(std::is_same_v<T, void>, "Unsupported metadata type");
        }
    }, val);
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
      read_binary(is, v);
      return v;
    }
    case 1:
    {
      double v;
      read_binary(is, v);
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
    mutable std::mutex index_mutex;

    Impl(const std::string &path, const Config &cfg) : db_path(path), config(cfg), space(cfg.vector_dim)
    {
      // use configured max_elements to initialize HNSW index
      hnsw_index = new hnswlib::HierarchicalNSW<float>(&space, static_cast<size_t>(config.max_elements), 16, 200, true);
    }
    ~Impl() { delete hnsw_index; }

    // helper to remove id from metadata inverted index
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

    // rebuild HNSW index with new capacity (must be called under index_mutex)
    bool rebuild_index(size_t new_max_elements)
    {
      // create a new index
      hnswlib::HierarchicalNSW<float> *new_index = nullptr;
      try
      {
        new_index = new hnswlib::HierarchicalNSW<float>(&space, new_max_elements, 16, 200, true);
        // add all vectors from storage
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
      // swap indexes
      delete hnsw_index;
      hnsw_index = new_index;
      config.max_elements = new_max_elements;
      return true;
    }

    bool save()
    {
      std::lock_guard<std::mutex> lock(index_mutex);

      // We'll write a temp DB file and the HNSW index to a temp file, fsync, then rename atomically.
      const std::string tmp_db = db_path + ".tmp";
      const std::string tmp_hnsw_path = db_path + ".tmp_hnsw";

      // save HNSW index separately first
      hnsw_index->saveIndex(tmp_hnsw_path);

      // serialize everything into the tmp_db
      std::ofstream ofs(tmp_db, std::ios::binary | std::ios::out | std::ios::trunc);
      if (!ofs)
        return false;

      // Magic + format version
      ofs.write("ORIONDB1", 8);
      uint32_t format_version = 1;
      write_binary(ofs, format_version);

      // --- CONFIG ---
      write_config(ofs, config);

      // --- STORAGE / VECTORS ---
      uint64_t storage_count = storage.size();
      write_binary(ofs, storage_count);
      for (const auto &kv : storage)
      {
        write_binary(ofs, kv.first);              // id
        // write vector
        const Vector &v = kv.second.vector;
        uint64_t vec_len = v.size();
        write_binary(ofs, vec_len);
        if (vec_len > 0)
          ofs.write(reinterpret_cast<const char *>(v.data()), static_cast<std::streamsize>(vec_len * sizeof(float)));
        // write metadata
        const Metadata &meta = kv.second.metadata;
        uint64_t meta_pairs = meta.size();
        write_binary(ofs, meta_pairs);
        for (const auto &m : meta)
        {
          write_string(ofs, m.first);
          write_metadata_value(ofs, m.second);
        }
      }

      // --- METADATA INDEX block ---
      // serialize inverted index to a memory buffer first to avoid partial writes
      std::stringstream meta_idx_stream(std::ios::binary | std::ios::in | std::ios::out);
      uint64_t outer_map_size = metadata_index.size();
      write_binary(meta_idx_stream, outer_map_size);
      for (const auto &outer : metadata_index)
      {
        write_string(meta_idx_stream, outer.first);
        uint64_t inner_map_size = outer.second.size();
        write_binary(meta_idx_stream, inner_map_size);
        for (const auto &inner : outer.second)
        {
          write_metadata_value(meta_idx_stream, inner.first);
          uint64_t id_set_size = inner.second.size();
          write_binary(meta_idx_stream, id_set_size);
          for (auto id : inner.second)
            write_binary(meta_idx_stream, id);
        }
      }
      meta_idx_stream.seekg(0, std::ios::end);
      uint64_t meta_idx_size = static_cast<uint64_t>(meta_idx_stream.tellg());
      meta_idx_stream.seekg(0, std::ios::beg);

      // write size + blob
      write_binary(ofs, meta_idx_size);
      std::string meta_idx_data;
      meta_idx_data.resize(static_cast<size_t>(meta_idx_size));
      meta_idx_stream.read(&meta_idx_data[0], static_cast<std::streamsize>(meta_idx_size));
      ofs.write(meta_idx_data.data(), static_cast<std::streamsize>(meta_idx_size));

      ofs.flush();

#if defined(_WIN32)
      // On Windows, ensure the file system has flushed buffers for the file.
      {
        HANDLE hFile = CreateFileA(tmp_db.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile != INVALID_HANDLE_VALUE)
        {
          if (!FlushFileBuffers(hFile))
          {
            std::cerr << "Warning: FlushFileBuffers() failed: GetLastError=" << GetLastError() << std::endl;
          }
          CloseHandle(hFile);
        }
        else
        {
          std::cerr << "Warning: CreateFileA failed for FlushFileBuffers: GetLastError=" << GetLastError() << std::endl;
        }
      }
#else
      int fd_sync = ::open(tmp_db.c_str(), O_RDONLY);
      if (fd_sync != -1)
      {
        if (::fsync(fd_sync) != 0)
        {
          std::cerr << "Warning: fsync() failed: errno=" << errno << std::endl;
        }
        ::close(fd_sync);
      }
      else
      {
        std::cerr << "Warning: open() failed for fsync: errno=" << errno << std::endl;
      }
#endif

      ofs.close();

      // rename tmp_hnsw -> final hnsw file (overwrite if exists)
      {
        const std::string final_hnsw = db_path + ".hnsw";
        if (std::rename(tmp_hnsw_path.c_str(), final_hnsw.c_str()) != 0)
        {
          std::cerr << "Warning: cannot rename HNSW tmp file: errno=" << errno << std::endl;
        }
      }

      if (std::rename(tmp_db.c_str(), db_path.c_str()) != 0)
      {
        std::cerr << "Error: Cannot atomically rename tmp DB file to final DB file: errno=" << errno << std::endl;
        return false;
      }

      return true;
    }

    bool load()
    {
      std::lock_guard<std::mutex> lock(index_mutex);

      std::ifstream ifs(db_path, std::ios::binary | std::ios::in);
      if (!ifs)
        return false;

      char magic[8];
      ifs.read(magic, 8);
      uint32_t format_version = 0;
      read_binary(ifs, format_version);
      if (std::memcmp(magic, "ORIONDB1", 8) != 0)
      {
        std::cerr << "Invalid DB magic." << std::endl;
        return false;
      }
      // --- CONFIG ---
      read_config(ifs, config);

      // reinitialize space/index according to loaded config
      // delete old index if exists
      if (hnsw_index)
      {
        delete hnsw_index;
        hnsw_index = nullptr;
      }
      space = hnswlib::L2Space(config.vector_dim);
      hnsw_index = new hnswlib::HierarchicalNSW<float>(&space, static_cast<size_t>(config.max_elements), 16, 200, true);

      // --- STORAGE ---
      uint64_t storage_count = 0;
      read_binary(ifs, storage_count);
      storage.clear();
      for (uint64_t i = 0; i < storage_count; ++i)
      {
        VectorId id;
        read_binary(ifs, id);
        uint64_t vec_len = 0;
        read_binary(ifs, vec_len);
        Vector v;
        v.resize(static_cast<size_t>(vec_len));
        if (vec_len > 0)
          ifs.read(reinterpret_cast<char *>(v.data()), static_cast<std::streamsize>(vec_len * sizeof(float)));
        uint64_t meta_pairs = 0;
        read_binary(ifs, meta_pairs);
        Metadata meta;
        for (uint64_t m = 0; m < meta_pairs; ++m)
        {
          std::string key = read_string(ifs);
          MetadataValue mv = read_metadata_value(ifs);
          meta.emplace(std::move(key), std::move(mv));
        }
        storage[id] = {v, meta};
      }

      // --- METADATA INDEX block ---
      metadata_index.clear();
      uint64_t meta_idx_size = 0;
      read_binary(ifs, meta_idx_size);
      if (meta_idx_size > 0)
      {
        std::vector<char> meta_idx_buffer(static_cast<size_t>(meta_idx_size));
        ifs.read(meta_idx_buffer.data(), static_cast<std::streamsize>(meta_idx_size));
        if (static_cast<uint64_t>(ifs.gcount()) != meta_idx_size)
          throw std::runtime_error("Failed to read metadata index block.");

        std::stringstream meta_idx_stream(std::ios::binary | std::ios::in | std::ios::out);
        meta_idx_stream.write(meta_idx_buffer.data(), static_cast<std::streamsize>(meta_idx_buffer.size()));
        meta_idx_stream.seekg(0, std::ios::beg);

        uint64_t outer_map_size = 0;
        read_binary(meta_idx_stream, outer_map_size);
        for (uint64_t i = 0; i < outer_map_size; ++i)
        {
          std::string outer_key = read_string(meta_idx_stream);
          uint64_t inner_map_size = 0;
          read_binary(meta_idx_stream, inner_map_size);
          for (uint64_t j = 0; j < inner_map_size; ++j)
          {
            MetadataValue mv = read_metadata_value(meta_idx_stream);
            uint64_t id_set_size = 0;
            read_binary(meta_idx_stream, id_set_size);
            std::set<VectorId> ids;
            for (uint64_t k = 0; k < id_set_size; ++k)
            {
              VectorId id;
              read_binary(meta_idx_stream, id);
              ids.insert(id);
            }
            metadata_index[outer_key][mv] = std::move(ids);
          }
        }
      }

      // load hnsw index file if present
      const std::string hnsw_path = db_path + ".hnsw";
      {
        std::ifstream hifs(hnsw_path, std::ios::binary | std::ios::in);
        if (hifs)
        {
          try
          {
            hnsw_index->loadIndex(hnsw_path, &space);
          }
          catch (const std::exception &e)
          {
            std::cerr << "Warning: failed to load HNSW index file: " << e.what() << std::endl;
            // rebuild index from storage below
          }
        }
      }

      // Ensure HNSW contains at least the elements from storage.
      // If loadIndex succeeded and had data, it's fine. Otherwise add points from storage.
      // We conservatively add all points — hnswlib will manage duplicates if load worked.
      for (const auto &kv : storage)
      {
        try
        {
          hnsw_index->addPoint(kv.second.vector.data(), kv.first);
        }
        catch (...)
        {
          // if capacity exceeded or other error -> ignore here; user should recreate DB with larger max_elements
        }
      }

      return true;
    }

    bool add(VectorId id, const Vector &vec, const Metadata &meta)
    {
      if (vec.size() != config.vector_dim)
        return false;
      std::lock_guard<std::mutex> lock(index_mutex);

      // If id already exists: remove from metadata index and mark as deleted in HNSW
      if (storage.count(id))
      {
        remove_from_metadata_index(id);
        try {
          hnsw_index->markDelete(id);
        } catch (...) {
          // hnswlib may throw if label not present; ignore and continue
        }
      }

      storage[id] = {vec, meta};

      // Attempt to add; if it fails due to capacity, rebuild with larger capacity and retry.
      try
      {
        hnsw_index->addPoint(vec.data(), id);
      }
      catch (const std::exception &e)
      {
        // decide new size: at least double, and at least storage.size()
        size_t proposed = std::max<size_t>(config.max_elements * 2, storage.size() + 10);
        bool rebuilt = rebuild_index(proposed);
        if (!rebuilt)
        {
          std::cerr << "Failed to rebuild HNSW index when adding id " << id << ": " << e.what() << std::endl;
          return false;
        }
        try
        {
          hnsw_index->addPoint(vec.data(), id);
        }
        catch (const std::exception &e2)
        {
          std::cerr << "Failed to add point even after rebuild: " << e2.what() << std::endl;
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
      std::lock_guard<std::mutex> lock(index_mutex);
      if (query_vec.size() != config.vector_dim || storage.empty())
        return {};
      auto result_queue = hnsw_index->searchKnn(query_vec.data(), n);
      std::vector<QueryResult> results;
      results.reserve(result_queue.size());
      while (!result_queue.empty())
      {
        results.push_back(orion::QueryResult{result_queue.top().second, result_queue.top().first});
        result_queue.pop();
      }
      std::reverse(results.begin(), results.end());
      return results;
    }

    std::vector<QueryResult> query(const Vector &query_vec, size_t n, const Metadata &filter) const
    {
      std::lock_guard<std::mutex> lock(index_mutex);
      if (filter.empty())
        return query(query_vec, n);
      std::set<VectorId> candidate_ids;
      bool first_filter_clause = true;
      for (const auto &[key, value] : filter)
      {
        auto it_key = metadata_index.find(key);
        if (it_key == metadata_index.end())
          return {};
        auto it_val = it_key->second.find(value);
        if (it_val == it_key->second.end())
          return {};
        const auto &ids_for_this_clause = it_val->second;
        if (first_filter_clause)
        {
          candidate_ids = ids_for_this_clause;
          first_filter_clause = false;
        }
        else
        {
          std::set<VectorId> intersection;
          std::set_intersection(candidate_ids.begin(), candidate_ids.end(),
                                ids_for_this_clause.begin(), ids_for_this_clause.end(),
                                std::inserter(intersection, intersection.begin()));
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
        bool operator()(hnswlib::labeltype current_id) override
        {
          return allowed_ids.count(current_id);
        }
      };

      IdFilterFunctor filter_functor(candidate_ids);
      auto result_queue = hnsw_index->searchKnn(query_vec.data(), n, &filter_functor);
      std::vector<QueryResult> results;
      results.reserve(result_queue.size());
      while (!result_queue.empty()) {
        results.push_back(orion::QueryResult{result_queue.top().second, result_queue.top().first});
        result_queue.pop();
      }
      std::reverse(results.begin(), results.end());
      return results;
    }

    std::optional<std::pair<Vector, Metadata>> get(VectorId id) const
    {
      std::lock_guard<std::mutex> lock(index_mutex);
      auto it = storage.find(id);
      if (it != storage.end())
        return std::make_pair(it->second.vector, it->second.metadata);
      return std::nullopt;
    }

    bool remove(VectorId id)
    {
      std::lock_guard<std::mutex> lock(index_mutex);
      if (storage.find(id) == storage.end())
        return false;
      remove_from_metadata_index(id);
      try
      {
        hnsw_index->markDelete(id);
      }
      catch (...)
      {
        // ignore
      }
      storage.erase(id);
      return true;
    }

    size_t count() const
    {
      std::lock_guard<std::mutex> lock(index_mutex);
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
  std::string Database::get_version() { return "0.1.0-alpha"; }

} // namespace orion
