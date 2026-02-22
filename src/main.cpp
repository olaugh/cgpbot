#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <dpp/dpp.h>

#include "board.h"

// ---------------------------------------------------------------------------
// Returns true if the filename looks like an image we should process.
// ---------------------------------------------------------------------------
static bool is_image(const std::string& filename) {
    static constexpr std::string_view exts[] = {".png", ".jpg", ".jpeg", ".webp"};
    for (auto ext : exts) {
        if (filename.size() >= ext.size() &&
            filename.compare(filename.size() - ext.size(), ext.size(), ext) == 0) {
            return true;
        }
    }
    return false;
}

int main() {
    const char* token_env = std::getenv("CGPBOT_TOKEN");
    if (!token_env) {
        std::cerr << "Set CGPBOT_TOKEN environment variable.\n";
        return 1;
    }

    dpp::cluster bot(token_env, dpp::i_default_intents | dpp::i_message_content);

    bot.on_log(dpp::utility::cout_logger());

    bot.on_ready([&bot](const dpp::ready_t&) {
        std::cout << "Logged in as " << bot.me.username << "\n";
    });

    bot.on_message_create([&bot](const dpp::message_create_t& event) {
        // Ignore our own messages.
        if (event.msg.author.id == bot.me.id) {
            return;
        }

        // Look for the first image attachment.
        for (const auto& att : event.msg.attachments) {
            if (!is_image(att.filename)) {
                continue;
            }

            // Download the image into memory (no disk I/O).
            bot.request(
                att.url,
                dpp::m_get,
                [&bot, channel_id = event.msg.channel_id,
                 msg_id = event.msg.id](const dpp::http_request_completion_t& res) {
                    if (res.status != 200) {
                        bot.message_create(dpp::message(channel_id,
                            "[error: failed to download attachment]")
                            .set_reference(msg_id));
                        return;
                    }

                    std::vector<uint8_t> buf(res.body.begin(), res.body.end());
                    std::string cgp = process_board_image(buf);

                    bot.message_create(dpp::message(channel_id,
                        "```\n" + cgp + "\n```")
                        .set_reference(msg_id));
                });

            // Process only the first image per message.
            break;
        }
    });

    bot.start(dpp::st_wait);
}
