import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import dotenv from "dotenv";
import { fileURLToPath } from "url";
import path from "path";
import { z } from "zod";

// Load .env from the telegram-mcp directory regardless of CWD
const __dirname = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.join(__dirname, "..", ".env") });

const BOT_TOKEN = process.env.TELEGRAM_BOT_TOKEN;
if (!BOT_TOKEN) {
  process.stderr.write("TELEGRAM_BOT_TOKEN is not set. Add it to telegram-mcp/.env\n");
  process.exit(1);
}

const BASE_URL = `https://api.telegram.org/bot${BOT_TOKEN}`;

async function telegramRequest(
  method: string,
  params: Record<string, unknown> = {}
): Promise<unknown> {
  const res = await fetch(`${BASE_URL}/${method}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });

  const data = (await res.json()) as {
    ok: boolean;
    result?: unknown;
    description?: string;
    error_code?: number;
  };

  if (!data.ok) {
    throw new Error(
      `Telegram API error ${data.error_code ?? "unknown"}: ${data.description}`
    );
  }

  return data.result;
}

const server = new McpServer({
  name: "telegram",
  version: "1.0.0",
});

// ── send_message ────────────────────────────────────────────────────────────
server.tool(
  "send_message",
  "Send a text message to a Telegram chat via the bot",
  {
    chat_id: z
      .union([z.string(), z.number()])
      .describe("Chat ID (numeric) or public @username"),
    text: z.string().min(1).describe("Message text (supports Markdown/HTML)"),
    parse_mode: z
      .enum(["Markdown", "MarkdownV2", "HTML"])
      .optional()
      .describe("Optional formatting mode"),
    disable_notification: z
      .boolean()
      .optional()
      .describe("Send silently (no sound/vibration)"),
  },
  async ({ chat_id, text, parse_mode, disable_notification }) => {
    const result = await telegramRequest("sendMessage", {
      chat_id,
      text,
      ...(parse_mode && { parse_mode }),
      ...(disable_notification !== undefined && { disable_notification }),
    });
    return {
      content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
    };
  }
);

// ── get_updates ─────────────────────────────────────────────────────────────
server.tool(
  "get_updates",
  "Fetch recent updates (messages, commands) received by the bot",
  {
    limit: z
      .number()
      .int()
      .min(1)
      .max(100)
      .default(10)
      .describe("Number of updates to return (1-100, default 10)"),
    offset: z
      .number()
      .int()
      .optional()
      .describe("Offset to mark earlier updates as read"),
    allowed_updates: z
      .array(z.string())
      .optional()
      .describe("Filter update types, e.g. [\"message\", \"callback_query\"]"),
  },
  async ({ limit, offset, allowed_updates }) => {
    const result = await telegramRequest("getUpdates", {
      limit,
      ...(offset !== undefined && { offset }),
      ...(allowed_updates && { allowed_updates }),
    });
    return {
      content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
    };
  }
);

// ── get_chat_info ────────────────────────────────────────────────────────────
server.tool(
  "get_chat_info",
  "Get details about a Telegram chat (type, title, member count, etc.)",
  {
    chat_id: z
      .union([z.string(), z.number()])
      .describe("Chat ID (numeric) or public @username"),
  },
  async ({ chat_id }) => {
    const [chat, memberCount] = await Promise.allSettled([
      telegramRequest("getChat", { chat_id }),
      telegramRequest("getChatMemberCount", { chat_id }),
    ]);

    const result = {
      chat: chat.status === "fulfilled" ? chat.value : { error: (chat as PromiseRejectedResult).reason?.message },
      member_count:
        memberCount.status === "fulfilled"
          ? memberCount.value
          : { error: (memberCount as PromiseRejectedResult).reason?.message },
    };

    return {
      content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
    };
  }
);

// ── boot ────────────────────────────────────────────────────────────────────
const transport = new StdioServerTransport();
await server.connect(transport);
