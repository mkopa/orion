#include "orion/database.h"
#include <gtest/gtest.h>
#include <filesystem>
#include <random>
#include <thread>
#include <vector>
#include <chrono>
#include <atomic>

using namespace orion;
namespace fs = std::filesystem;

static std::vector<float> random_vector(size_t dim, std::mt19937 &rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(dim);
    for (size_t i = 0; i < dim; ++i) v[i] = dist(rng);
    return v;
}

TEST(SerializationAndRebuild, SaveLoadAndRebuild)
{
    // prepare temp path
    fs::path tmp = fs::temp_directory_path() / "orion_test_db.bin";
    std::error_code ec;
    fs::remove(tmp, ec);

    const uint32_t dim = 8;
    Config cfg(dim, 4); // small max_elements to force rebuild
    auto created = Database::create(tmp.string(), cfg);
    ASSERT_TRUE(created.has_value());
    Database db = std::move(created.value());

    std::mt19937 rng(12345);

    const int total = 50; // more than max_elements to force rebuild
    for (int i = 0; i < total; ++i) {
        Vector v = random_vector(dim, rng);
        Metadata meta;
        meta["i"] = int64_t(i);
        if (i % 2 == 0) meta["tag"] = std::string("even");
        else meta["tag"] = std::string("odd");
        float score = float(i) * 0.5;
        meta["score"] = double(score);
        bool ok = db.add(static_cast<VectorId>(i+1), v, meta);
        ASSERT_TRUE(ok);
    }

    ASSERT_EQ(db.count(), static_cast<size_t>(total));

    // save
    ASSERT_TRUE(db.save());

    // load into a new instance
    auto loaded_opt = Database::load(tmp.string());
    ASSERT_TRUE(loaded_opt.has_value());
    Database loaded = std::move(loaded_opt.value());
    ASSERT_EQ(loaded.count(), static_cast<size_t>(total));

    // check a few random elements for correctness
    for (int check_id : {1, 2, 10, 25, 49}) {
        auto got = loaded.get(static_cast<VectorId>(check_id));
        ASSERT_TRUE(got.has_value());
        auto [vec, meta] = got.value();
        ASSERT_EQ(vec.size(), dim);
        ASSERT_TRUE(meta.find("i") != meta.end());
        ASSERT_TRUE(std::holds_alternative<int64_t>(meta["i"]));
        int64_t orig = std::get<int64_t>(meta["i"]);
        ASSERT_EQ(orig, check_id - 1);
    }

    // cleanup
    fs::remove(tmp, ec);
}

TEST(Concurrency, ParallelAddAndQuery)
{
    fs::path tmp = fs::temp_directory_path() / "orion_test_db2.bin";
    std::error_code ec;
    fs::remove(tmp, ec);

    const uint32_t dim = 16;
    Config cfg(dim, 128);
    auto created = Database::create(tmp.string(), cfg);
    ASSERT_TRUE(created.has_value());
    Database db = std::move(created.value());

    const int threads = 6;
    const int per_thread = 200;
    std::atomic<int> added{0};

    auto producer = [&](int tid) {
        std::mt19937 rng(1000 + tid);
        for (int i = 0; i < per_thread; ++i) {
            int id = tid * per_thread + i + 1;
            Vector v = random_vector(dim, rng);
            Metadata m;
            m["thread"] = int64_t(tid);
            m["seq"] = int64_t(i);
            bool ok = db.add(static_cast<VectorId>(id), v, m);
            if (ok) added.fetch_add(1, std::memory_order_relaxed);
            // occasionally query
            if ((i & 0x1F) == 0) {
                Vector q = random_vector(dim, rng);
                auto res = db.query(q, 5);
                (void)res;
            }
        }
    };

    std::vector<std::thread> ths;
    for (int t = 0; t < threads; ++t) ths.emplace_back(producer, t);
    for (auto &t : ths) t.join();

    ASSERT_EQ(added.load(), threads * per_thread);
    ASSERT_EQ(db.count(), static_cast<size_t>(threads * per_thread));

    // quick queries after everything
    Vector q(dim, 0.1f);
    auto res = db.query(q, 10);
    (void)res;

    // cleanup
    fs::remove(tmp, ec);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}