#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <variant>
#include <map>
#include <optional>

namespace orion {

using VectorId = uint64_t;
using Vector = std::vector<float>;
using MetadataValue = std::variant<int64_t, double, std::string>;
using Metadata = std::map<std::string, MetadataValue>;

struct QueryResult
{
    VectorId id;
    float distance;
};

struct Config
{
    uint32_t vector_dim = 0;
    uint64_t max_elements = 1000000; // default max elements for HNSW index

    Config() = default;
    Config(uint32_t dim, uint64_t max_elems = 1000000) : vector_dim(dim), max_elements(max_elems) {}
};

class Database
{
public:
    // create a new database at path (overwrites if exists)
    static std::optional<Database> create(const std::string &path, const Config &config);
    // load an existing database from path
    static std::optional<Database> load(const std::string &path);

    Database(Database &&other) noexcept;
    Database &operator=(Database &&other) noexcept;
    ~Database();

    // save current in-memory DB to disk (atomic)
    bool save();

    // add or update a vector with metadata
    bool add(VectorId id, const Vector &vec, const Metadata &meta);

    // query top-n nearest neighbors (no filter)
    std::vector<QueryResult> query(const Vector &query_vec, size_t n) const;

    // query top-n nearest neighbors with metadata filter (AND of key=value pairs)
    std::vector<QueryResult> query(const Vector &query_vec, size_t n, const Metadata &filter) const;

    // retrieve raw vector and metadata
    std::optional<std::pair<Vector, Metadata>> get(VectorId id) const;

    // remove a vector by id
    bool remove(VectorId id);

    // number of stored vectors
    size_t count() const;

    // version string
    static std::string get_version();

private:
    Database();
    class Impl;
    Impl *pimpl;
};

} // namespace orion
