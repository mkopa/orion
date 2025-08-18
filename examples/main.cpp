#include <iostream>
#include <vector>
#include <cstdio>
#include <random>
#include <chrono>
#include "orion/database.h"

void print_results(const std::string& title, const std::vector<orion::QueryResult>& results) {
    std::cout << "--- " << title << " ---" << std::endl;
    if (results.empty()) {
        std::cout << "No results found." << std::endl;
        return;
    }
    for (const auto& res : results) {
        std::cout << "ID: " << res.id << ", Distance: " << res.distance << std::endl;
    }
}

void run_persistence_test() {
    const std::string db_path = "persistence_test.orion";
    
    // --- STAGE 1: Create, populate, and SAVE the database ---
    std::cout << "\n=== STAGE 1: Creating, populating, and saving ===" << std::endl;
    {
        orion::Config config;
        config.vector_dim = 2;
        auto db_optional = orion::Database::create(db_path, config);
        if (!db_optional) {
            std::cerr << "CRITICAL FAILURE: Could not create database!" << std::endl;
            return;
        }
        auto db = std::move(*db_optional);

        db.add(1, {0.1f, 0.1f}, {{"type", "animal"}, {"color", "red"}});
        db.add(2, {0.2f, 0.2f}, {{"type", "plant"}, {"color", "green"}});
        db.add(3, {0.9f, 0.9f}, {{"type", "animal"}, {"color", "blue"}});
        
        std::cout << "DB contains " << db.count() << " vectors. Saving..." << std::endl;
        if (!db.save()) {
            std::cerr << "FAILURE: Could not save database!" << std::endl;
        }
    }

    // --- STAGE 2: LOAD the database and verify its state ---
    std::cout << "\n=== STAGE 2: Loading and verifying ===" << std::endl;
    {
        auto db_optional = orion::Database::load(db_path);
        if (!db_optional) {
            std::cerr << "FAILURE: Failed to load DB!" << std::endl;
            return;
        }
        auto db = std::move(*db_optional);

        std::cout << "Loaded DB contains " << db.count() << " vectors." << std::endl;
        if (db.count() != 3) {
            std::cerr << "FAILURE: Vector count mismatch!" << std::endl;
            return;
        }
        
        orion::Vector query_vec = {0.8f, 0.8f};
        orion::Metadata filter = {{"type", "animal"}, {"color", "blue"}};
        auto results = db.query(query_vec, 1, filter);
        
        print_results("Query with filter on loaded DB", results);

        if (results.size() == 1 && results[0].id == 3) {
            std::cout << "SUCCESS: Filtering works correctly on the loaded database." << std::endl;
        } else {
            std::cout << "FAILURE: Incorrect filter results on the loaded database." << std::endl;
        }
    }
}

void run_large_scale_test() {
    const std::string db_path = "large_test.orion";
    std::remove(db_path.c_str());
    std::remove((db_path + ".hnsw").c_str());

    orion::Config cfg(32, 100000);
    auto db_opt = orion::Database::create(db_path, cfg);
    if (!db_opt) {
        std::cerr << "Cannot create DB\n";
        return;
    }
    auto db = std::move(*db_opt);

    std::mt19937 rng(123);
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < 100000; ++i) {
        orion::Vector v(32);
        for (auto &f : v) f = std::uniform_real_distribution<float>(-1, 1)(rng);
        orion::Metadata m;
        m["id"] = int64_t(i);
        db.add(i, v, m);
    }

    auto mid = std::chrono::high_resolution_clock::now();
    std::cout << "Inserted 100k vectors in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(mid - start).count()
              << " ms\n";

    orion::Vector q(32, 0.5f);
    auto results = db.query(q, 5);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Query returned " << results.size() << " results in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - mid).count()
              << " ms\n";

    db.save();
}

void run_extreme_scale_test() {
    const std::string db_path = "extreme_test.orion";
    std::remove(db_path.c_str());
    std::remove((db_path + ".hnsw").c_str());

    const size_t total = 1000000; // 1 mln
    const uint32_t dim = 32;

    orion::Config cfg(dim, total);
    auto db_opt = orion::Database::create(db_path, cfg);
    if (!db_opt) {
        std::cerr << "Cannot create DB for extreme test\n";
        return;
    }
    auto db = std::move(*db_opt);

    std::mt19937 rng(999);
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < total; ++i) {
        orion::Vector v(dim);
        for (auto &f : v) f = std::uniform_real_distribution<float>(-1, 1)(rng);
        orion::Metadata m;
        m["id"] = int64_t(i);
        db.add(i, v, m);

        if ((i % 100000) == 0 && i > 0) {
            std::cout << "Inserted " << i << " vectors...\n";
        }
    }

    auto mid = std::chrono::high_resolution_clock::now();
    std::cout << "Inserted 1M vectors in "
              << std::chrono::duration_cast<std::chrono::seconds>(mid - start).count()
              << " s\n";

    orion::Vector q(dim, 0.5f);
    auto results = db.query(q, 10);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Query returned " << results.size() << " results in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - mid).count()
              << " ms\n";

    db.save();
}

int main() {
    const std::string db_path = "persistence_test.orion";
    std::remove(db_path.c_str());
    
    run_persistence_test();
    run_large_scale_test();

    // WARNING: Very heavy test – run only if you know what you’re doing.
    // run_extreme_scale_test();

    std::remove(db_path.c_str());
    return 0;
}
