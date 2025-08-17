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

#include "hnswlib/hnswlib.h"

namespace orion
{

  template <typename T>
  void write_binary(std::ostream &os, const T &value) { os.write(reinterpret_cast<const char *>(&value), sizeof(T)); }
  template <typename T>
  void read_binary(std::istream &is, T &value) { is.read(reinterpret_cast<char *>(&value), sizeof(T)); }
  void write_string(std::ostream &os, const std::string &str)
  {
    uint64_t len = str.size();
    write_binary(os, len);
    os.write(str.c_str(), len);
  }
  std::string read_string(std::istream &is)
  {
    uint64_t len;
    read_binary(is, len);
    if (len == 0)
      return "";
    std::string str(len, '\0');
    is.read(&str[0], len);
    return str;
  }
  void write_metadata_value(std::ostream &os, const MetadataValue &val);
  MetadataValue read_metadata_value(std::istream &is);
  void write_metadata_value(std::ostream &os, const MetadataValue &val)
  {
    uint8_t type_index = val.index();
    write_binary(os, type_index);
    std::visit([&os](auto &&arg)
               {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, double>) write_binary(os, arg);
        else if constexpr (std::is_same_v<T, std::string>) write_string(os, arg); }, val);
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
      throw std::runtime_error("Invalid variant type index during deserialization.");
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
      hnsw_index = new hnswlib::HierarchicalNSW<float>(&space, 1000000, 16, 200, true);
    }
    ~Impl() { delete hnsw_index; }

    bool save()
    {
      std::lock_guard<std::mutex> lock(index_mutex);

      std::string tmp_hnsw_path = db_path + ".tmp_hnsw";
      hnsw_index->saveIndex(tmp_hnsw_path);
      std::ifstream tmp_ifs(tmp_hnsw_path, std::ios::binary | std::ios::ate);
      if (!tmp_ifs)
      {
        std::cerr << "Error: Cannot create temporary HNSW file for saving." << std::endl;
        return false;
      }
      auto tmp_size = static_cast<size_t>(tmp_ifs.tellg());
      tmp_ifs.seekg(0, std::ios::beg);
      std::vector<char> hnsw_data(tmp_size);
      if (tmp_size)
        tmp_ifs.read(hnsw_data.data(), hnsw_data.size());
      tmp_ifs.close();
      std::remove(tmp_hnsw_path.c_str());

      std::stringstream storage_stream(std::ios::binary | std::ios::in | std::ios::out);
      write_binary(storage_stream, (uint64_t)storage.size());
      for (const auto &[id, data] : storage)
      {
        write_binary(storage_stream, id);
        write_binary(storage_stream, (uint64_t)data.vector.size());
        storage_stream.write(reinterpret_cast<const char *>(data.vector.data()), data.vector.size() * sizeof(float));
        write_binary(storage_stream, (uint64_t)data.metadata.size());
        for (const auto &[meta_key, meta_val] : data.metadata)
        {
          write_string(storage_stream, meta_key);
          write_metadata_value(storage_stream, meta_val);
        }
      }
      std::string storage_data = storage_stream.str();

      std::stringstream meta_idx_stream(std::ios::binary | std::ios::in | std::ios::out);
      write_binary(meta_idx_stream, (uint64_t)metadata_index.size());
      for (const auto &[key, inner_map] : metadata_index)
      {
        write_string(meta_idx_stream, key);
        write_binary(meta_idx_stream, (uint64_t)inner_map.size());
        for (const auto &[val, id_set] : inner_map)
        {
          write_metadata_value(meta_idx_stream, val);
          write_binary(meta_idx_stream, (uint64_t)id_set.size());
          for (const auto &id : id_set)
            write_binary(meta_idx_stream, id);
        }
      }
      std::string meta_idx_data = meta_idx_stream.str();

      std::string tmp_db = db_path + ".tmp";
      std::ofstream ofs(tmp_db, std::ios::binary | std::ios::trunc);
      if (!ofs)
      {
        std::cerr << "Error: Cannot open temp file for writing: " << tmp_db << std::endl;
        return false;
      }

      ofs.write("ORIONDB1", 8);
      write_binary(ofs, config);
      write_binary(ofs, (uint64_t)hnsw_data.size());
      write_binary(ofs, (uint64_t)storage_data.size());
      write_binary(ofs, (uint64_t)meta_idx_data.size());
      if (!hnsw_data.empty())
        ofs.write(hnsw_data.data(), hnsw_data.size());
      if (!storage_data.empty())
        ofs.write(storage_data.data(), storage_data.size());
      if (!meta_idx_data.empty())
        ofs.write(meta_idx_data.data(), meta_idx_data.size());

      ofs.flush();

#if defined(_WIN32)
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

      if (std::rename(tmp_db.c_str(), db_path.c_str()) != 0)
      {
        std::cerr << "Error: Cannot rename temp DB file to final path: " << tmp_db << " -> " << db_path << std::endl;
        std::remove(tmp_db.c_str());
        return false;
      }

      std::cout << "Database saved to " << db_path << std::endl;
      return true;
    }

    bool load()
    {
      std::lock_guard<std::mutex> lock(index_mutex);

      std::ifstream ifs(db_path, std::ios::binary);
      if (!ifs)
      {
        return false;
      }

      // --- header ---
      char magic[8];
      ifs.read(magic, 8);
      if (static_cast<std::streamsize>(ifs.gcount()) != 8 || std::string(magic, 8) != "ORIONDB1")
      {
        throw std::runtime_error("Invalid Orion DB file format or file is empty.");
      }

      // read config
      ifs.read(reinterpret_cast<char *>(&config), sizeof(config));
      if (!ifs.good())
        throw std::runtime_error("Failed to read config from DB file.");

      uint64_t hnsw_size = 0, storage_size = 0, meta_idx_size = 0;
      ifs.read(reinterpret_cast<char *>(&hnsw_size), sizeof(hnsw_size));
      ifs.read(reinterpret_cast<char *>(&storage_size), sizeof(storage_size));
      ifs.read(reinterpret_cast<char *>(&meta_idx_size), sizeof(meta_idx_size));
      if (!ifs.good())
        throw std::runtime_error("Failed to read sizes header from DB file.");

      // --- HNSW block ---
      if (hnsw_size > 0)
      {
        std::vector<char> hnsw_data(hnsw_size);
        ifs.read(hnsw_data.data(), static_cast<std::streamsize>(hnsw_size));
        if (static_cast<uint64_t>(ifs.gcount()) != hnsw_size)
          throw std::runtime_error("Failed to read HNSW block.");

        std::string tmp_hnsw_path = db_path + ".tmp_hnsw";
        {
          std::ofstream tmp_ofs(tmp_hnsw_path, std::ios::binary | std::ios::trunc);
          if (!tmp_ofs)
            throw std::runtime_error("Failed to create temporary HNSW file for loading.");
          tmp_ofs.write(hnsw_data.data(), static_cast<std::streamsize>(hnsw_data.size()));
          tmp_ofs.close();
        }

        // rebuild space and load index
        space = hnswlib::L2Space(config.vector_dim);
        delete hnsw_index;
        hnsw_index = new hnswlib::HierarchicalNSW<float>(&space, 1, 1, 1, true);
        try
        {
          hnsw_index->loadIndex(tmp_hnsw_path, &space);
        }
        catch (const std::exception &e)
        {
          std::remove(tmp_hnsw_path.c_str());
          throw;
        }
        std::remove(tmp_hnsw_path.c_str());
      }
      else
      {
        space = hnswlib::L2Space(config.vector_dim);
        delete hnsw_index;
        hnsw_index = new hnswlib::HierarchicalNSW<float>(&space, 1, 1, 1, true);
      }

      // --- STORAGE block ---
      storage.clear();
      if (storage_size > 0)
      {
        std::vector<char> storage_buffer(storage_size);
        ifs.read(storage_buffer.data(), static_cast<std::streamsize>(storage_size));
        if (static_cast<uint64_t>(ifs.gcount()) != storage_size)
          throw std::runtime_error("Failed to read storage block.");

        std::stringstream storage_stream(std::ios::binary | std::ios::in | std::ios::out);
        storage_stream.write(storage_buffer.data(), static_cast<std::streamsize>(storage_buffer.size()));
        storage_stream.seekg(0, std::ios::beg);

        uint64_t storage_count = 0;
        read_binary(storage_stream, storage_count);
        if (!storage_stream.good())
          throw std::runtime_error("Failed to read storage count.");

        for (uint64_t i = 0; i < storage_count; ++i)
        {
          VectorId id;
          read_binary(storage_stream, id);
          if (!storage_stream.good())
            throw std::runtime_error("Failed to read VectorId in storage.");

          uint64_t vec_len;
          read_binary(storage_stream, vec_len);
          if (!storage_stream.good())
            throw std::runtime_error("Failed to read vector length in storage.");

          Vector vec(vec_len);
          if (vec_len > 0)
          {
            storage_stream.read(reinterpret_cast<char *>(vec.data()), static_cast<std::streamsize>(vec_len * sizeof(float)));
            if (!storage_stream.good())
              throw std::runtime_error("Failed to read vector data in storage.");
          }

          uint64_t meta_count;
          read_binary(storage_stream, meta_count);
          if (!storage_stream.good())
            throw std::runtime_error("Failed to read metadata count in storage.");

          Metadata meta;
          for (uint64_t j = 0; j < meta_count; ++j)
          {
            std::string key = read_string(storage_stream);
            if (!storage_stream.good())
              throw std::runtime_error("Failed to read metadata key in storage.");
            MetadataValue val = read_metadata_value(storage_stream);
            if (!storage_stream.good())
              throw std::runtime_error("Failed to read metadata value in storage.");
            meta[key] = val;
          }

          storage[id] = {vec, meta};
        }
      }

      // --- METADATA INDEX block ---
      metadata_index.clear();
      if (meta_idx_size > 0)
      {
        std::vector<char> meta_idx_buffer(meta_idx_size);
        ifs.read(meta_idx_buffer.data(), static_cast<std::streamsize>(meta_idx_size));
        if (static_cast<uint64_t>(ifs.gcount()) != meta_idx_size)
          throw std::runtime_error("Failed to read metadata index block.");

        std::stringstream meta_idx_stream(std::ios::binary | std::ios::in | std::ios::out);
        meta_idx_stream.write(meta_idx_buffer.data(), static_cast<std::streamsize>(meta_idx_buffer.size()));
        meta_idx_stream.seekg(0, std::ios::beg);

        uint64_t outer_map_size = 0;
        read_binary(meta_idx_stream, outer_map_size);
        if (!meta_idx_stream.good())
          throw std::runtime_error("Failed to read metadata index outer map size.");

        for (uint64_t i = 0; i < outer_map_size; ++i)
        {
          std::string key = read_string(meta_idx_stream);
          if (!meta_idx_stream.good())
            throw std::runtime_error("Failed to read metadata index key.");
          uint64_t inner_map_size = 0;
          read_binary(meta_idx_stream, inner_map_size);
          if (!meta_idx_stream.good())
            throw std::runtime_error("Failed to read metadata index inner map size.");

          for (uint64_t j = 0; j < inner_map_size; ++j)
          {
            MetadataValue val = read_metadata_value(meta_idx_stream);
            if (!meta_idx_stream.good())
              throw std::runtime_error("Failed to read metadata index value.");
            uint64_t set_size = 0;
            read_binary(meta_idx_stream, set_size);
            if (!meta_idx_stream.good())
              throw std::runtime_error("Failed to read metadata index set size.");

            for (uint64_t k = 0; k < set_size; ++k)
            {
              VectorId id;
              read_binary(meta_idx_stream, id);
              if (!meta_idx_stream.good())
                throw std::runtime_error("Failed to read VectorId in metadata index.");
              metadata_index[key][val].insert(id);
            }
          }
        }
      }

      std::cout << "Database loaded from " << db_path << ". Vectors: " << storage.size() << std::endl;
      return true;
    }

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
    bool add(VectorId id, const Vector &vec, const Metadata &meta)
    {
      if (vec.size() != config.vector_dim)
        return false;
      std::lock_guard<std::mutex> lock(index_mutex);
      if (storage.count(id))
        remove_from_metadata_index(id);
      storage[id] = {vec, meta};
      hnsw_index->addPoint(vec.data(), id);
      for (const auto &[key, value] : meta)
      {
        metadata_index[key][value].insert(id);
      }
      return true;
    }
    std::vector<QueryResult> query(const Vector &query_vec, size_t n, const Metadata &filter) const
    {
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
      while (!result_queue.empty())
      {
        results.push_back(orion::QueryResult{result_queue.top().second, result_queue.top().first});
        result_queue.pop();
      }
      std::reverse(results.begin(), results.end());
      return results;
    }
    std::vector<QueryResult> query(const Vector &query_vec, size_t n) const
    {
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
    bool remove(VectorId id)
    {
      std::lock_guard<std::mutex> lock(index_mutex);
      if (storage.count(id))
      {
        remove_from_metadata_index(id);
        storage.erase(id);
        hnsw_index->markDelete(id);
        return true;
      }
      return false;
    }
    std::optional<std::pair<Vector, Metadata>> get(VectorId id) const
    {
      std::lock_guard<std::mutex> lock(index_mutex);
      auto it = storage.find(id);
      if (it != storage.end())
        return std::make_pair(it->second.vector, it->second.metadata);
      return std::nullopt;
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
      Database db;
      db.pimpl = new Impl(path, config);
      std::cout << "Creating new database at " << path << std::endl;
      return db;
    }
    catch (const std::exception &e)
    {
      std::cerr << "Error during DB creation: " << e.what() << std::endl;
      return std::nullopt;
    }
  }
  std::optional<Database> Database::load(const std::string &path)
  {
    try
    {
      std::ifstream test_ifs(path);
      if (!test_ifs.good() || test_ifs.peek() == EOF)
      {
        return std::nullopt;
      }
      Database db;
      db.pimpl = new Impl(path, {});
      if (db.pimpl->load())
      {
        return db;
      }
      delete db.pimpl;
      db.pimpl = nullptr;
      return std::nullopt;
    }
    catch (const std::exception &e)
    {
      std::cerr << "Error during DB loading: " << e.what() << std::endl;
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