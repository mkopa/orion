 # Orion
 
 ### The SQLite for AI
 

 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
 ![Language](https://img.shields.io/badge/language-C%2B%2B20-blue.svg)
 ![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)
 
 **Orion** is a high-performance, lightweight, embedded vector database engine written in modern C++20. It was conceived to address a growing need in the AI landscape: the ability to perform complex, low-latency vector operations directly within an application, without the overhead and dependencies of a client-server architecture.

Traditional vector databases excel at managing massive, distributed indexes in the cloud. Orion, however, is engineered for a different paradigm, bringing the power of large-scale similarity search to the edge. By operating as a linkable library, it eliminates network latency, ensures data privacy, and enables applications to function entirely offline. This makes Orion the ideal solution for on-device AI, edge computing, and any system where data locality, speed, and reliability are paramount.

At its core, Orion's performance is built upon two key algorithmic foundations:
HNSW for Similarity Search: For its Approximate Nearest Neighbor (ANN) search capabilities, Orion implements the Hierarchical Navigable Small World (HNSW) algorithm. Instead of performing a brute-force search with linear complexity O(N), HNSW constructs a multi-layered graph of interconnected data points. Queries traverse this graph from a sparse top layer (analogous to a highway system) to progressively denser lower layers (local roads), achieving a near-logarithmic complexity of O(log N). This allows for querying millions of vectors in milliseconds, providing an exceptional balance between search speed and accuracy.

Inverted Index for Metadata Filtering: Recognizing that vector similarity is often just one dimension of a query, Orion integrates a robust metadata filtering system. This is powered by a classic Inverted Index, a data structure that maps metadata key-value pairs to the set of vector IDs possessing them. When a query includes a filter (e.g., "find vectors similar to X where category = 'A' and active = true"), Orion first performs a rapid set intersection operation on the inverted index to produce a small candidate set of allowed vector IDs. Only then is the computationally intensive HNSW search performed on this pre-filtered set. This two-phase approach dramatically prunes the search space, enabling complex, multi-faceted queries to execute with minimal performance impact.

With C++20, Orion leverages modern language features for memory safety and a clean, expressive API, delivering a tool that is not only powerful but also a pleasure to integrate and use.
 
 ---
 
 ### Core Philosophy & Key Features
 
 Orion is built on a set of core principles that differentiate it from large, server-based vector databases.
 
 *   **🚀 Truly Embedded:** Orion is a library, not a server. Link it directly to your application for zero-latency data access. There are no network calls, no REST APIs, just pure function calls.
 
 *   **⚡ High-Performance ANN Search:** At its core, Orion uses a highly optimized implementation of HNSW (Hierarchical Navigable Small World), one of the fastest and most accurate algorithms for Approximate Nearest Neighbor search.
 
 *   **🗂️ Rich Metadata Filtering:** Don't just search by similarity. Orion features a powerful pre-filtering mechanism that allows you to narrow down your search space based on metadata *before* the vector search occurs, leading to highly relevant and efficient queries.
 
 *   **📦 Single-File Database:** The entire database, including vectors, HNSW index, and metadata indexes, is stored in a single, portable file, making backups and data management trivial.
 
 *   **🛡️ Modern C++ Design:** Built with C++20, Orion leverages modern language features for performance, safety, and a clean, expressive API. The design uses idioms like Pimpl to ensure a stable and clean public interface.
 
 *   **💾 Atomic & Crash-Safe Saves:** Explicit `save()` operations ensure that the database file is written atomically, protecting against data corruption in case of an application crash.
 
 ### When to use Orion
 
 Orion excels in scenarios where data locality and low latency are critical.
 
 **Ideal use cases:**
 *   **On-device AI:** Mobile or desktop apps that need to perform similarity search on local data (e.g., photo matching, semantic search in notes).
 *   **IoT & Edge Computing:** Devices that need to analyze sensor data or images locally without relying on a cloud connection.
 *   **Application-Specific Backends:** Services that require a fast, local cache or index for a subset of data (e.g., recommendation engines for a user's immediate context).
 
 **When *not* to use Orion:**
 *   As a replacement for petabyte-scale, distributed, cloud-native vector databases like Pinecone or Weaviate. Orion is designed for a single-node, embedded context.
 
 ---
 
 ### Quick Start
 
 Here is a minimal example demonstrating the full lifecycle of creating, populating, saving, loading, and querying an Orion database.
 
 ```cpp
 #include <iostream>
 #include <vector>
 #include <cstdio>
 #include "orion/database.h"
 
 int main() {
     const std::string db_path = "my_app_db.orion";
     
     // 1. Create a new database
     {
         orion::Config config;
         config.vector_dim = 2; // Our vectors will have 2 dimensions
         auto db_optional = orion::Database::create(db_path, config);
         if (!db_optional) {
             return 1; // Failed to create
         }
         auto db = std::move(*db_optional);
 
         // 2. Add data with metadata
         db.add(1, {1.0f, 2.0f}, {{"category", "A"}, {"active", (int64_t)1}});
         db.add(2, {1.1f, 2.1f}, {{"category", "A"}, {"active", (int64_t)0}});
         db.add(3, {5.0f, 6.0f}, {{"category", "B"}, {"active", (int64_t)1}});
 
         // 3. Save the database to disk
         db.save();
     }
 
     // 4. Load the database from disk
     {
         auto db_optional = orion::Database::load(db_path);
         if (!db_optional) {
             return 1; // Failed to load
         }
         auto db = std::move(*db_optional);
 
         // 5. Perform a filtered query
         // "Find the vector closest to {5.2, 6.2} that has category 'B' and is active"
         orion::Vector query_vec = {5.2f, 6.2f};
         orion::Metadata filter = {{"category", "B"}, {"active", (int64_t)1}};
         
         auto results = db.query(query_vec, 1, filter);
 
         // 6. Print results (should find ID 3)
         for (const auto& res : results) {
             std::cout << "Found ID: " << res.id 
                       << ", Distance: " << res.distance << std::endl;
         }
     }
     
     std::remove(db_path.c_str());
     return 0;
 }
 ```
 
 ### Building the Project
 
 Orion uses CMake for building. The `hnswlib` dependency is included as a git submodule.
 
 ```bash
 # 1. Clone the repository recursively to fetch submodules
 git clone --recursive https://github.com/mkopa/orion.git
 cd orion
 
 # 2. Configure the project using CMake
 mkdir build
 cd build
 cmake ..
 
 # 3. Build the library and examples
 cmake --build .
 
 # 4. Run the example
 ./examples/hello_orion
 ```
 
 ### API Reference
 
 The primary interface is the `orion::Database` class.
 
 *   `static std::optional<Database> create(path, config)`: Creates a new, empty database file.
 *   `static std::optional<Database> load(path)`: Loads an existing database from a file.
 *   `bool save()`: Atomically saves the current state of the database to its file.
 *   `bool add(id, vector, metadata)`: Adds or updates a vector with its associated metadata.
 *   `std::vector<QueryResult> query(vector, k)`: Finds the `k` nearest neighbors.
 *   `std::vector<QueryResult> query(vector, k, filter)`: Finds the `k` nearest neighbors among vectors that match all metadata filter conditions.
 *   `bool remove(id)`: Deletes a vector from the database.
 *   `std::optional<...> get(id)`: Retrieves a vector and its metadata by ID.
 *   `size_t count()`: Returns the number of vectors in the database.
 
 ### Project Status
 
 **Alpha.** The core API is functional and the primary features (add, query with filters, save, load) are implemented. However, the API may still evolve. It is not yet recommended for production use without thorough testing in your specific environment.
 
 ### Contributing
 
 Contributions are welcome! Please feel free to open an issue to discuss a bug or a new feature, or submit a pull request with your changes.
 
 ### License
 
 Orion is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.