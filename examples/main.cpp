#include <iostream>
#include <vector>
#include <cstdio>
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

int main() {
    const std::string db_path = "persistence_test.orion";
    std::remove(db_path.c_str());
    
    run_persistence_test();

    std::remove(db_path.c_str());
    return 0;
}