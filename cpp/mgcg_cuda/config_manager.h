#pragma once

#include <fstream>
#include "nlohmann/json.hpp"



class ConfigManager {
public:
    ConfigManager() {
        // Load the configuration file
        std::filesystem::path source_path(__FILE__);
        std::filesystem::path source_dir = source_path.parent_path();
        std::filesystem::path config_file = source_dir / "amgconfig.json";
        
        std::ifstream f(config_file);
        if (!f.is_open()) {
            std::cerr << "No such file!" << std::endl;
            exit(1);
        }
        data = nlohmann::json::parse(f);
    }

    // Get the value of a key in the configuration file
    template <typename T>
    T get(const std::string& key) {
        return data[key].get<T>();
    }


    nlohmann::json data;
};

