#pragma once

#include <string>
#include <vector>

namespace ufra {

// Utility functions
std::string getLibraryVersion();
std::vector<std::string> splitString(const std::string& str, char delimiter);
bool fileExists(const std::string& path);

} // namespace ufra