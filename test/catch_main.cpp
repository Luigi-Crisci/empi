#include <catch2/catch_session.hpp>
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>


int main(int argc, char *argv[]) {
    Catch::Session session;

    using namespace Catch::Clara;
    const auto cli = session.cli();

    session.cli(cli);

    int return_code = session.applyCommandLine(argc, argv);
    if(return_code != 0) { return return_code; }

    return_code = session.run();
    return return_code;
}

struct global_setup_and_teardown : Catch::EventListenerBase {
    using EventListenerBase::EventListenerBase;
    void testCasePartialEnded(const Catch::TestCaseStats &, uint64_t) override {
        // Reset REQUIRE_LOOP after each test case, section or generator value.
        celerity::test_utils::require_loop_assertion_registry::get_instance().reset();
    }
};

CATCH_REGISTER_LISTENER(global_setup_and_teardown);