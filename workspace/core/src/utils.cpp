#include "ufra/utils.h"
#include <sstream>
#include <fstream>

namespace ufra {

std::string getLibraryVersion() {
    return "1.0.0-minimal";
}

std::vector<std::string> splitString(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    
    return tokens;
}

bool fileExists(const std::string& path) {
    std::ifstream file(path);
    return file.good();
}

} // namespace ufra