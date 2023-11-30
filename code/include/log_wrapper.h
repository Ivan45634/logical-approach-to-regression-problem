#ifndef INCLUDE_LOG_WRAPPER_H_
#define INCLUDE_LOG_WRAPPER_H_

#include <boost/move/utility_core.hpp>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/sources/global_logger_storage.hpp>
#include <boost/log/sources/logger.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>

#define LOG_(x) BOOST_LOG_TRIVIAL(x)

enum SeverityType {
    kTrace = 0,
    kDebug = 1,
    kInfo = 2,
    kWarning = 3,
    kError = 4,
    kFatal = 5
};

namespace logging = boost::log;
namespace sinks = boost::log::sinks;
namespace src = boost::log::sources;
namespace expr = boost::log::expressions;
namespace attrs = boost::log::attributes;
namespace keywords = boost::log::keywords;


void init_logging(SeverityType sev_level=kInfo) {
    logging::register_simple_formatter_factory<logging::trivial::severity_level, char>("Severity");
    logging::add_file_log(keywords::file_name = "%Y-%m-%d.log",
                          keywords::auto_flush = true,
                          keywords::format = "(%TimeStamp%)[%Severity%]: %Message%");
    auto sev_filter = logging::trivial::info;

    switch(sev_level) {
        case kTrace:
            sev_filter = logging::trivial::trace;
            break;

        case kDebug:
            sev_filter = logging::trivial::debug;
            break;

        case kInfo:
            sev_filter = logging::trivial::info;
            break;

        case kWarning:
            sev_filter = logging::trivial::warning;
            break;

        case kError:
            sev_filter = logging::trivial::error;
            break;

        case kFatal:
            sev_filter = logging::trivial::fatal;
            break;

        default:
            break;
    }

    logging::core::get()->set_filter(logging::trivial::severity >= sev_filter);
    logging::add_common_attributes();
}

#endif  // INCLUDE_LOG_WRAPPER_H_
