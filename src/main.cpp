#include <exception>
#include <iostream>
#include <ostream>

#include "application.h"

int main() {
    try {
        application app;
        app.run();
    } catch (std::exception &e) {
        std::cout << "Caught exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}