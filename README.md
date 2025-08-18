 # Orion

 ### The SQLite for AI

 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
 ![Language](https://img.shields.io/badge/language-C%2B%2B20-blue.svg)
 ![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)

 **Orion** is a high-performance, lightweight, embedded vector database engine written in modern **C++20**.  
 It is designed for **on-device AI**, **edge computing**, and **low-latency similarity search** without the overhead of a client–server system.

 Unlike distributed cloud vector databases, Orion works as a **linkable library**.  
 There are no sockets, no RPCs – just direct function calls.  
 This makes it ideal for scenarios where **data locality, privacy, and offline operation** are essential.

 ---

 ## 🔑 Key Features

 - 🚀 **Embedded, not server-based** – link directly into your app.
 - ⚡ **Fast ANN search** – based on HNSW (Hierarchical Navigable Small World).
 - 🗂 **Rich metadata filtering** – inverted index allows pre-filtering before similarity search.
 - 📦 **Single-file database** – vectors, HNSW index, and metadata stored together.
 - 💾 **Atomic saves** – `save()` guarantees crash safety.
 - 🛡 **Modern C++ API** – clean design with Pimpl idiom, expressive, and stable.

 ---

 ## 📚 Architecture Overview

 - **HNSW for similarity search**  
   Enables approximate nearest neighbor search with near-logarithmic complexity O(log N), making queries over millions of vectors possible in milliseconds.

 - **Inverted index for metadata**  
   Enables filtering queries (e.g., *find nearest neighbors where `category=A` and `active=true`*) by pre-selecting candidate vectors before running similarity search.

 ---

 ## 🚦 When to Use Orion

 ✅ **Great for:**
 - On-device AI (photo search, semantic notes, recommender systems).
 - IoT / edge devices (offline analysis of local data).
 - Application-specific backends (fast local index/cache).

 ❌ **Not for:**
 - Distributed, petabyte-scale deployments.  
   Orion is single-node and embedded.

 ---

  ## 🆚 Comparison with Other Systems

 | Feature / System          | **Orion** (this project) | **SQLite** (relational DB) | **Pinecone** (cloud vector DB) |
 |---------------------------|--------------------------|----------------------------|--------------------------------|
 | Deployment Model          | Embedded library (link directly) | Embedded library (link directly) | Managed cloud service |
 | Data Model                | Vectors + Metadata       | Tables (rows & columns)    | Vectors + Metadata             |
 | ANN Search (HNSW)         | ✅ Yes                   | ❌ No                      | ✅ Yes                         |
 | Metadata Filtering        | ✅ Inverted Index        | ✅ SQL WHERE clauses       | ✅ Metadata filters            |
 | Scale                     | Local device / single-node | Local device / single-node | Distributed, multi-tenant      |
 | Offline Usage             | ✅ Full                  | ✅ Full                    | ❌ Requires internet           |
 | Latency                   | Ultra-low (in-process)   | Low (in-process SQL)       | Higher (network + RPC)         |
 | Storage Format            | Single `.orion` file     | Single `.sqlite` file      | Cloud-managed storage          |
 | Language Bindings         | C++20 (planned Python, Rust, Go) | Dozens (C, Python, Rust, etc.) | Python, JavaScript, REST APIs  |
 | Best Use Cases            | On-device AI, edge apps, semantic search | Relational data, app state | Large-scale distributed AI search |
 | License                   | MIT                      | Public domain              | Proprietary SaaS               |

 Orion aims to fill the same niche that **SQLite does for SQL**, but for **vector similarity search**:  
 lightweight, portable, and embedded — running **everywhere** without servers or cloud dependencies.


 ## ⚡ Quick Start

 ```cpp
 #include <iostream>
 #include "orion/database.h"

 int main() {
     const std::string db_path = "my_app_db.orion";

     // Create database
     orion::Config cfg;
     cfg.vector_dim = 2;
     auto db_opt = orion::Database::create(db_path, cfg);
     if (!db_opt) return 1;
     auto db = std::move(*db_opt);

     // Add data
     db.add(1, {1.0f, 2.0f}, {{"category", "A"}, {"active", (int64_t)1}});
     db.add(2, {5.0f, 6.0f}, {{"category", "B"}, {"active", (int64_t)1}});
     db.save();

     // Reload database
     auto loaded_opt = orion::Database::load(db_path);
     if (!loaded_opt) return 1;
     auto loaded = std::move(*loaded_opt);

     // Query with filter
     orion::Vector query = {5.1f, 6.2f};
     orion::Metadata filter = {{"category", "B"}, {"active", (int64_t)1}};
     auto results = loaded.query(query, 1, filter);

     for (auto& r : results) {
         std::cout << "Found ID: " << r.id 
                   << ", Distance: " << r.distance << "\n";
     }
 }
 ```

 ---

 ## 🛠 Building

 ```bash
 # Clone with submodules
 git clone --recursive https://github.com/mkopa/orion.git
 cd orion

 # Configure
 mkdir build && cd build
 cmake -DCMAKE_BUILD_TYPE=Release ..

 # Build library, examples, and tests
 cmake --build . -j
 ```

 Run example:

 ```bash
 ./examples/hello_orion
 ```

 ---

 ## 🧪 Running Tests

 Orion uses **GoogleTest**. Tests include concurrency safety and serialization checks.

 ```bash
 # Build tests
 cmake --build build --target orion_tests -j

 # Run directly
 ./build/tests/orion_tests

 # Or via CTest
 ctest --test-dir build --output-on-failure
 ```

 ---

 ## 💾 File Format & Versioning

 - Orion databases are **single-file** (`.orion` extension).  
 - Each database contains:
   - Vector data
   - HNSW graph index
   - Metadata inverted index
   - Internal config header (dimension, max_elements, version, etc.)

 ### Migration Guide
 - **Backward compatibility**: newer Orion can usually open older files.  
 - **Forward compatibility**: older Orion versions may *not* read files created by newer versions.  
 - Use the `export → import` API to migrate data across incompatible versions:
   ```cpp
   // Export old DB to JSON/CSV + vectors
   db.export("dump.json");

   // Import into a fresh DB with new Orion
   auto db2 = Database::create("new.orion", new_cfg);
   db2->import("dump.json");
   db2->save();
   ```

 ---

 ## 📖 API Reference

 - `Database::create(path, config)` – create a new DB.
 - `Database::load(path)` – open existing DB.
 - `bool save()` – atomically persist to disk.
 - `bool add(id, vector, metadata)` – add or update entry.
 - `std::optional<Entry> get(id)` – fetch by ID.
 - `bool remove(id)` – delete by ID.
 - `size_t count()` – number of entries.
 - `query(vec, k)` – nearest neighbors.
 - `query(vec, k, filter)` – nearest neighbors with metadata filter.

 ---

 ## 🚀 Advanced Features / Roadmap

 Orion is in active development. Planned and experimental features include:

 - 🔄 **Automatic index rebuild** – triggered when capacity is exceeded.
 - 🔍 **Hybrid queries** – combine full-text, metadata, and vector search.
 - 🧩 **Bindings for Python/Rust/Go** – simplify integration in non-C++ projects.
 - 🖥️ **GPU acceleration (CUDA/Metal)** – offload ANN search for higher throughput.
 - 🌐 **Optional RPC server mode** – run Orion in lightweight client/server setups.
 - 📤 **Streaming export/import** – incremental backup and restore.
 - 📊 **Monitoring hooks** – expose metrics for observability.
 - 🔐 **Encrypted storage** – protect database files at rest.
 - 🧪 **Fuzz & stress testing** – continuous reliability validation.

 Community feedback will shape which features are prioritized.

 ---

 ## 📌 Project Status

 **Alpha.**  
 Stable enough for experimentation, but the API may evolve.  
 Not production-ready yet.

 ---

 ## 🤝 Contributing

 Contributions are welcome!  
 - Report issues / feature requests via GitHub Issues.  
 - Submit pull requests with tests.  

 ---

 ## 📜 License

 Orion is licensed under the **MIT License**.  
 See the [LICENSE](LICENSE) file for details.
