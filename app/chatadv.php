<?php
declare(strict_types=1);

header("Content-Type: application/json; charset=utf-8");

/**
 * Minimal .env loader (no composer).
 * Loads KEY=VALUE lines into $_ENV and getenv().
 */
function load_dotenv(string $path): void {
    if (!is_file($path)) {
        return;
    }

    $lines = file($path, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
    if ($lines === false) {
        return;
    }

    foreach ($lines as $line) {
        $line = trim($line);
        if ($line === "" || str_starts_with($line, "#")) {
            continue;
        }

        $pos = strpos($line, "=");
        if ($pos === false) {
            continue;
        }

        $key = trim(substr($line, 0, $pos));
        $value = trim(substr($line, $pos + 1));

        // Strip surrounding quotes if present
        if ((str_starts_with($value, '"') && str_ends_with($value, '"')) ||
            (str_starts_with($value, "'") && str_ends_with($value, "'"))) {
            $value = substr($value, 1, -1);
        }

        $_ENV[$key] = $value;
        putenv($key . "=" . $value);
    }
}

function json_error(int $status, string $message): void {
    http_response_code($status);
    echo json_encode(["error" => $message], JSON_UNESCAPED_UNICODE);
    exit;
}

function openai_post_json(string $url, array $payload, string $apiKey): array {
    $ch = curl_init($url);

    curl_setopt_array($ch, [
        CURLOPT_POST => true,
        CURLOPT_HTTPHEADER => [
            "Authorization: Bearer {$apiKey}",
            "Content-Type: application/json",
        ],
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_POSTFIELDS => json_encode($payload, JSON_UNESCAPED_UNICODE),
        CURLOPT_TIMEOUT => 60,
    ]);

    $raw = curl_exec($ch);
    $status = (int) curl_getinfo($ch, CURLINFO_HTTP_CODE);

    if ($raw === false) {
        $err = curl_error($ch);
        curl_close($ch);
        throw new RuntimeException("cURL error: {$err}");
    }
    curl_close($ch);

    $data = json_decode($raw, true);

    if ($status >= 400) {
        $msg = is_array($data) ? json_encode($data, JSON_UNESCAPED_UNICODE) : $raw;
        throw new RuntimeException("OpenAI HTTP {$status}: {$msg}");
    }

    if (!is_array($data)) {
        throw new RuntimeException("Invalid JSON from OpenAI: {$raw}");
    }

    return $data;
}

// Load .env from the same directory
load_dotenv(__DIR__ . "/.env");

$apiKey = getenv("OPENAI_API_KEY") ?: "";
$vectorStoreId = getenv("VECTOR_STORE_ID") ?: "";
$model = getenv("OPENAI_MODEL") ?: "gpt-5.2-2025-12-11";

if ($apiKey === "") {
    json_error(500, "Missing OPENAI_API_KEY in .env");
}
if ($vectorStoreId === "") {
    json_error(500, "Missing VECTOR_STORE_ID in .env");
}

$systemPromptPath = __DIR__ . "/system_prompt.txt";
$systemPrompt = file_get_contents($systemPromptPath);
if ($systemPrompt === false || trim($systemPrompt) === "") {
    json_error(500, "Missing or empty system_prompt.txt");
}

// Read request body
$raw = file_get_contents("php://input");
$body = $raw ? json_decode($raw, true) : null;

if (!is_array($body)) {
    json_error(400, "Send JSON body like {\"message\":\"...\",\"previous_response_id\":\"...\"}");
}

$userMessage = $body["message"] ?? null;
$previousResponseId = $body["previous_response_id"] ?? null;

if (!is_string($userMessage) || trim($userMessage) === "") {
    json_error(400, "Field 'message' is required and must be a non-empty string.");
}
if ($previousResponseId !== null && !is_string($previousResponseId)) {
    json_error(400, "Field 'previous_response_id' must be a string if provided.");
}

// Build Responses API payload
$payload = [
    "model" => $model,
    "instructions" => $systemPrompt,
    "input" => [
        ["role" => "user", "content" => $userMessage]
    ],
    "tools" => [
        [
            "type" => "file_search",
            "vector_store_ids" => [$vectorStoreId],
        ]
    ],
    // Helpful while testing. Remove later if you do not want retrieved chunks returned.
    "include" => ["file_search_call.results"],
];

// Multi-turn continuity
if (is_string($previousResponseId) && $previousResponseId !== "") {
    $payload["previous_response_id"] = $previousResponseId;
}

try {
    $resp = openai_post_json("https://api.openai.com/v1/responses", $payload, $apiKey);

    // Return only what your UI needs
    function extract_output_text(array $resp): ?string {
        // Some responses include output_text at top-level in some SDKs/versions
        if (isset($resp["output_text"]) && is_string($resp["output_text"]) && $resp["output_text"] !== "") {
            return $resp["output_text"];
        }

        // Otherwise, extract from resp["output"] messages
        if (!isset($resp["output"]) || !is_array($resp["output"])) {
            return null;
        }

        $texts = [];

        foreach ($resp["output"] as $item) {
            if (!is_array($item)) {
                continue;
            }
            if (($item["type"] ?? null) !== "message") {
                continue;
            }
            $content = $item["content"] ?? null;
            if (!is_array($content)) {
                continue;
            }
            foreach ($content as $c) {
                if (!is_array($c)) {
                    continue;
                }
                // Most common: {"type":"output_text","text":"..."}
                if (($c["type"] ?? null) === "output_text" && isset($c["text"]) && is_string($c["text"])) {
                    $texts[] = $c["text"];
                }
            }
        }

        if (count($texts) === 0) {
            return null;
        }

        return implode("\n", $texts);
    }

    $assistantText = extract_output_text($resp);

    echo json_encode([
        "previous_response_id" => $resp["id"] ?? null,
        "text" => $assistantText,
        // keep for local debugging; remove in production
        "debug" => $resp,
    ], JSON_UNESCAPED_UNICODE);


} catch (Throwable $e) {
    json_error(500, $e->getMessage());
}
