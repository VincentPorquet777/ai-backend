import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { z } from "zod";

const BOT_TOKEN =
  process.env.TELEGRAM_BOT_TOKEN ||
  "8677946801:AAE52n6T3aHNwAGWpOVJxs2GAjCM4WhLJGQ";

const BASE_URL = `https://api.telegram.org/bot${BOT_TOKEN}`;

async function telegramRequest(method, params = {}) {
  const res = await fetch(`${BASE_URL}/${method}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  const data = await res.json();
  if (!data.ok) {
    throw new Error(`Telegram error ${data.error_code ?? "?"}: ${data.description}`);
  }
  return data.result;
}

function createServer() {
  const server = new McpServer({ name: "telegram", version: "1.0.0" });

  server.tool(
    "send_message",
    "Send a text message to a Telegram chat via the bot",
    {
      chat_id: z.union([z.string(), z.number()]).describe("Chat ID or @username"),
      text: z.string().min(1).describe("Message text (supports Markdown/HTML)"),
      parse_mode: z.enum(["Markdown", "MarkdownV2", "HTML"]).optional(),
      disable_notification: z.boolean().optional(),
    },
    async ({ chat_id, text, parse_mode, disable_notification }) => {
      const result = await telegramRequest("sendMessage", {
        chat_id,
        text,
        ...(parse_mode && { parse_mode }),
        ...(disable_notification !== undefined && { disable_notification }),
      });
      return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }] };
    }
  );

  server.tool(
    "get_updates",
    "Fetch recent messages and commands received by the bot",
    {
      limit: z.number().int().min(1).max(100).default(10),
      offset: z.number().int().optional(),
    },
    async ({ limit, offset }) => {
      const result = await telegramRequest("getUpdates", {
        limit,
        ...(offset !== undefined && { offset }),
      });
      return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }] };
    }
  );

  server.tool(
    "get_chat_info",
    "Get details and member count for a Telegram chat",
    {
      chat_id: z.union([z.string(), z.number()]).describe("Chat ID or @username"),
    },
    async ({ chat_id }) => {
      const [chat, memberCount] = await Promise.allSettled([
        telegramRequest("getChat", { chat_id }),
        telegramRequest("getChatMemberCount", { chat_id }),
      ]);
      const result = {
        chat: chat.status === "fulfilled" ? chat.value : { error: chat.reason?.message },
        member_count:
          memberCount.status === "fulfilled"
            ? memberCount.value
            : { error: memberCount.reason?.message },
      };
      return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }] };
    }
  );

  return server;
}

export default async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization, Accept, Mcp-Session-Id");

  if (req.method === "OPTIONS") {
    res.status(200).end();
    return;
  }

  const transport = new StreamableHTTPServerTransport({
    sessionIdGenerator: undefined, // stateless — safe for serverless
  });

  const server = createServer();
  await server.connect(transport);
  await transport.handleRequest(req, res, req.body);
}
