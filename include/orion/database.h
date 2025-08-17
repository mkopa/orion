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

struct Config {
    uint32_t vector_dim;
};

struct QueryResult {
    VectorId id;
    float distance;
};

class Database {
public:
    // Factory methods for clear object lifecycle
    static std::optional<Database> create(const std::string& path, const Config& config);
    static std::optional<Database> load(const std::string& path);

    // Rule of Five for proper resource management
    Database(const Database&) = delete;
    Database& operator=(const Database&) = delete;
    Database(Database&&) noexcept;
    Database& operator=(Database&&) noexcept;
    ~Database();

    // Core functionalities
    bool save();
    bool add(VectorId id, const Vector& vec, const Metadata& meta);
    std::vector<QueryResult> query(const Vector& query_vec, size_t n) const;
    std::vector<QueryResult> query(const Vector& query_vec, size_t n, const Metadata& filter) const;
    std::optional<std::pair<Vector, Metadata>> get(VectorId id) const;
    bool remove(VectorId id);
    size_t count() const;
    static std::string get_version();

private:
    Database();
    class Impl;
    Impl* pimpl;
};

} // namespace orion