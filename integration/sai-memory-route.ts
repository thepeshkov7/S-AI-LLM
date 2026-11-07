// Place at app/api/chat/sai-memory/route.ts in the S AI server checkout.
import { getAuthenticatedContext } from "@/lib/auth";
import { isTrustedMutation, noStoreJson } from "@/lib/http";

export const runtime = "nodejs";

type AccountMemory = { memory_key: string; value: string };

function words(text: string) {
  return text.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? [];
}

const recallIntents: { questions: RegExp; keys: string[] }[] = [
  { questions: /^(?:как меня зовут|какое у меня имя|напомни моё имя|what is my name|remind me of my name)$/,
    keys: ["name", "profile name", "first name", "имя", "имя пользователя"] },
  { questions: /^(?:где я живу|в каком городе я живу|напомни мой город|какой город я называю домом|where do i live|which city do i live in|what city do i call home|remind me of my city)$/,
    keys: ["city", "profile city", "home city", "город", "город проживания"] },
  { questions: /^(?:какой у меня любимый цвет|какой цвет я люблю|напомни мой любимый цвет|what is my favou?rite colou?r|which colou?r do i like)$/,
    keys: ["favorite color", "preferences favorite color", "preferred colour", "любимый цвет"] },
  { questions: /^(?:какая у меня любимая еда|какое блюдо я люблю|напомни моё любимое блюдо|what is my favou?rite food|which dish do i like)$/,
    keys: ["favorite food", "preferences favorite food", "favorite dish", "любимая еда", "любимое блюдо"] },
  { questions: /^(?:как зовут моего питомца|напомни кличку питомца|what is my pet s name|remind me what my pet is called)$/,
    keys: ["pet name", "profile pet name", "name of pet", "кличка питомца"] },
  { questions: /^(?:на каком языке я предпочитаю общаться|напомни мой язык общения|which language do i prefer to chat in|what language should replies use)$/,
    keys: ["preferred language", "preferences language", "reply language", "язык общения"] },
  { questions: /^(?:как называется мой проект|над каким проектом я работаю|напомни мой текущий проект|what is my project called|which project am i working on)$/,
    keys: ["current project", "work project", "project name", "текущий проект", "название проекта"] },
  { questions: /^(?:какой спорт я люблю|напомни мой любимый вид спорта|what sport do i like|which sport is my favou?rite)$/,
    keys: ["favorite sport", "preferences sport", "preferred sport", "любимый спорт"] },
];

function recallKeys(question: string) {
  const normalized = words(question).join(" ");
  return recallIntents.find((intent) => intent.questions.test(normalized))?.keys;
}

export async function POST(request: Request) {
  if (process.env.SAI_MEMORY_CHAT_ENABLED !== "true") {
    return noStoreJson({ error: "not_found" }, { status: 404 });
  }
  if (!isTrustedMutation(request)) {
    return noStoreJson({ error: "forbidden" }, { status: 403 });
  }
  let payload: { question?: unknown; language?: unknown; useMemory?: unknown };
  try {
    const decoded: unknown = await request.json();
    if (!decoded || typeof decoded !== "object" || Array.isArray(decoded)) {
      return noStoreJson({ error: "invalid_request" }, { status: 400 });
    }
    payload = decoded;
  } catch {
    return noStoreJson({ error: "invalid_request" }, { status: 400 });
  }
  const question = typeof payload.question === "string" ? payload.question.trim() : "";
  const language = payload.language;
  if (!question || question.length > 1000 || (language !== "ru" && language !== "en") ||
      (payload.useMemory !== undefined && typeof payload.useMemory !== "boolean")) {
    return noStoreJson({ error: "invalid_request" }, { status: 400 });
  }
  try {
    const account = await getAuthenticatedContext(request);
    if (!account) return noStoreJson({ error: "unauthorized" }, { status: 401 });
    const keys = payload.useMemory === true ? recallKeys(question) : undefined;
    if (keys) {
      const { data, error } = await account.supabase
        .from("memories")
        .select("memory_key,value")
        .eq("user_id", account.user.id)
        .not("memory_key", "like", "settings.%")
        .order("updated_at", { ascending: false })
        .limit(100);
      if (error) return noStoreJson({ error: "memory_unavailable" }, { status: 503 });
      const matching = ((data ?? []) as AccountMemory[]).filter(
        (row) => keys.includes(words(row.memory_key).join(" ")) && row.value.trim(),
      );
      const values = [...new Set(matching.map((row) => row.value))];
      if (!values.length) return noStoreJson({ error: "memory_not_found" }, { status: 404 });
      if (values.length > 1) return noStoreJson({ error: "memory_ambiguous" }, { status: 409 });
      return noStoreJson({ completion: values[0], source: "account_memory", language });
    }
    const token = process.env.SAI_MEMORY_INFERENCE_TOKEN;
    const url = new URL(process.env.SAI_MEMORY_INFERENCE_URL ?? "");
    if (!token || url.protocol !== "https:" || url.username || url.password) {
      return noStoreJson({ error: "model_unavailable" }, { status: 503 });
    }
    const response = await fetch(new URL("/chat", url), {
      method: "POST",
      headers: { "Authorization": `Bearer ${token}`, "Content-Type": "application/json" },
      body: JSON.stringify({ question, language }),
      cache: "no-store",
      redirect: "error",
      signal: AbortSignal.any([request.signal, AbortSignal.timeout(15000)]),
    });
    if (!response.ok) {
      return noStoreJson({ error: response.status === 429 ? "rate_limited" : "model_unavailable" },
                         { status: response.status === 429 ? 429 : 503 });
    }
    const result = await response.json();
    if (result?.source === "model" && typeof result.completion === "string" && result.completion.trim()) {
      return noStoreJson({ completion: result.completion, source: result.source, language });
    }
    return noStoreJson({ error: "model_unavailable" }, { status: 503 });
  } catch {
    return noStoreJson({ error: "model_unavailable" }, { status: 503 });
  }
}
