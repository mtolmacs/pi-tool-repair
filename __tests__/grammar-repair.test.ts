import { describe, expect, it } from "vitest";
import {
  normalizeGrammarRepairConfig,
  parseToolGrammarLeaks,
  repairAssistantMessageGrammarLeaks,
  resolveGrammarRepairForModel,
  type GrammarRepairConfig,
  type MinimalAssistantMessage,
} from "../src/index.js";

const enabledConfig: GrammarRepairConfig = {
  enabled: true,
  grammars: [
    "dsml",
    "invoke",
    "qwen",
    "kimi",
    "mistral",
    "llama",
    "glm",
    "granite",
    "minimax-text",
    "olmo",
  ],
  mode: "recover",
  requireKnownTool: true,
  debug: false,
};

describe("grammar leak parsing", () => {
  it("parses DeepSeek DSML double-bar, single-bar, and no-lead-bar variants", () => {
    const text = `
<｜｜DSML｜｜tool_calls>
<｜｜DSML｜｜invoke name="code_exec">
<｜｜DSML｜｜parameter name="language" string="true">python</｜｜DSML｜｜parameter>
</｜｜DSML｜｜invoke>
</｜｜DSML｜｜tool_calls>
<｜DSML｜tool_calls>
<｜DSML｜invoke name="fetch">
<｜DSML｜parameter name="url" string="false">["x"]</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>
<DSML｜tool_calls>
<DSML｜invoke name="bash">
</DSML｜invoke>`;

    const calls = parseToolGrammarLeaks(text, ["dsml"]);
    expect(calls).toEqual([
      { grammar: "dsml", name: "code_exec", arguments: { language: "python" } },
      { grammar: "dsml", name: "fetch", arguments: { url: ["x"] } },
      { grammar: "dsml", name: "bash", arguments: {} },
    ]);
  });

  it("preserves newlines inside DSML parameter values", () => {
    const text = `<｜DSML｜tool_calls>
<｜DSML｜invoke name="bash">
<｜DSML｜parameter name="command" string="true">echo one
&& echo two</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>`;

    const [call] = parseToolGrammarLeaks(text, ["dsml"]);
    expect(call?.arguments).toEqual({ command: "echo one\n&& echo two" });
  });

  it("parses MiniMax invoke/parameter XML", () => {
    const text = `<minimax:tool_call>
<invoke name="search_web">
<parameter name="query_list">["weather"]</parameter>
</invoke>
</minimax:tool_call>`;

    const calls = parseToolGrammarLeaks(text, ["invoke"]);
    expect(calls).toEqual([
      { grammar: "invoke", name: "search_web", arguments: { query_list: ["weather"] } },
    ]);
  });

  it("parses Qwen function/parameter XML", () => {
    const text = `<tool_call>
<function=get_weather>
<parameter=location>Paris</parameter>
</function>
</tool_call>`;

    const calls = parseToolGrammarLeaks(text, ["qwen"]);
    expect(calls).toEqual([
      { grammar: "qwen", name: "get_weather", arguments: { location: "Paris" } },
    ]);
  });

  it("parses Kimi sentinel tool calls", () => {
    const text = `<|tool_calls_section_begin|><|tool_call_begin|>functions.web_search:0<|tool_call_argument_begin|>{"query":"pi"}<|tool_call_end|><|tool_calls_section_end|>`;

    const calls = parseToolGrammarLeaks(text, ["kimi"]);
    expect(calls).toEqual([
      { grammar: "kimi", name: "web_search", arguments: { query: "pi" } },
    ]);
  });

  it("keeps multiple calls from the same wrapper range", () => {
    const text = `<|tool_calls_section_begin|><|tool_call_begin|>functions.first:0<|tool_call_argument_begin|>{"a":1}<|tool_call_end|><|tool_call_begin|>functions.second:1<|tool_call_argument_begin|>{"b":2}<|tool_call_end|><|tool_calls_section_end|>`;

    const calls = parseToolGrammarLeaks(text, ["kimi"]);
    expect(calls).toEqual([
      { grammar: "kimi", name: "first", arguments: { a: 1 } },
      { grammar: "kimi", name: "second", arguments: { b: 2 } },
    ]);
  });

  it("parses Mistral TOOL_CALLS JSON", () => {
    const text = `[TOOL_CALLS] [{"name":"calculator","arguments":{"operation":"2+2"},"id":"abc123XYZ"}]`;

    const calls = parseToolGrammarLeaks(text, ["mistral"]);
    expect(calls).toEqual([
      { grammar: "mistral", name: "calculator", arguments: { operation: "2+2" } },
    ]);
  });

  it("parses bare Mistral JSON tool text", () => {
    const text = `{"name":"calculator","arguments":{"operation":"2+2"}}`;

    const calls = parseToolGrammarLeaks(text, ["mistral"]);
    expect(calls).toEqual([
      { grammar: "mistral", name: "calculator", arguments: { operation: "2+2" } },
    ]);
  });

  it("parses Llama python_tag JSON", () => {
    const text = `<|python_tag|>{"name":"write_file","arguments":{"path":"/tmp/a","content":"x"}}`;

    const calls = parseToolGrammarLeaks(text, ["llama"]);
    expect(calls).toEqual([
      { grammar: "llama", name: "write_file", arguments: { path: "/tmp/a", content: "x" } },
    ]);
  });

  it("parses bare Llama JSON tool text", () => {
    const text = `{"name":"write_file","arguments":{"path":"/tmp/a","content":"x"}}`;

    const calls = parseToolGrammarLeaks(text, ["llama"]);
    expect(calls).toEqual([
      { grammar: "llama", name: "write_file", arguments: { path: "/tmp/a", content: "x" } },
    ]);
  });

  it("parses GLM arg_key/arg_value XML", () => {
    const text = `<tool_call>get_weather
<arg_key>city</arg_key>
<arg_value>Beijing</arg_value>
</tool_call>`;

    const calls = parseToolGrammarLeaks(text, ["glm"]);
    expect(calls).toEqual([
      { grammar: "glm", name: "get_weather", arguments: { city: "Beijing" } },
    ]);
  });

  it("parses GLM zero-argument XML", () => {
    const text = `<tool_call>get_current_date</tool_call>`;

    const calls = parseToolGrammarLeaks(text, ["glm"]);
    expect(calls).toEqual([
      { grammar: "glm", name: "get_current_date", arguments: {} },
    ]);
  });

  it("parses Granite JSON tool_call", () => {
    const text = `<tool_call>
{"name":"get_current_weather","arguments":{"city":"London"}}
</tool_call>`;

    const calls = parseToolGrammarLeaks(text, ["granite"]);
    expect(calls).toEqual([
      { grammar: "granite", name: "get_current_weather", arguments: { city: "London" } },
    ]);
  });

  it("parses Granite pythonic tool text", () => {
    const text = `get_weather(location="San Francisco", unit="celsius")`;

    const calls = parseToolGrammarLeaks(text, ["granite"]);
    expect(calls).toEqual([
      { grammar: "granite", name: "get_weather", arguments: { location: "San Francisco", unit: "celsius" } },
    ]);
  });

  it("parses MiniMax-Text-01 typescript function calls", () => {
    const text = `<function_call>\`\`\`typescript
functions.get_current_weather({"location":"Shanghai"})
\`\`\``;

    const calls = parseToolGrammarLeaks(text, ["minimax-text"]);
    expect(calls).toEqual([
      { grammar: "minimax-text", name: "get_current_weather", arguments: { location: "Shanghai" } },
    ]);
  });

  it("parses OLMo pythonic function calls", () => {
    const text = `<function_calls>
write_file(path="/tmp/a", content="hello", overwrite=True)
</function_calls>`;

    const calls = parseToolGrammarLeaks(text, ["olmo"]);
    expect(calls).toEqual([
      { grammar: "olmo", name: "write_file", arguments: { path: "/tmp/a", content: "hello", overwrite: true } },
    ]);
  });

  it("does not parse tool grammar inside markdown code fences", () => {
    const text = "```xml\n<tool_call>{\"name\":\"bash\",\"arguments\":{}}</tool_call>\n```";
    expect(parseToolGrammarLeaks(text, ["granite"])).toEqual([]);
  });
});

// Truncated / dangling DSML markers — stream died mid-token. These can never
// be recovered as tool calls (incomplete), but the raw marker should not
// persist as visible assistant text. Covered for issue #3712.
describe("DSML dangling marker stripping", () => {
  it("does not report a truncated DSML open marker as a recovered call", () => {
    expect(parseToolGrammarLeaks("I'll read the file.\n<｜DSML｜tool_calls", ["dsml"])).toEqual([]);
  });

  it("does not report orphan markers from a truncated body as recovered calls", () => {
    expect(parseToolGrammarLeaks("<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"read\">", ["dsml"])).toEqual([]);
  });

  it("parses spaced DSML markers (space between prefix and tag name)", () => {
    const text = `Let me run the tests.

<｜DSML｜ tool_calls>
<｜DSML｜ invoke name="bash">
<｜DSML｜ parameter name="command" string="true">cd /tmp && ls</｜DSML｜ parameter>
<｜DSML｜ parameter name="timeout" string="false">600</｜DSML｜ parameter>
</｜DSML｜ invoke>
</｜DSML｜ calls>`;

    const calls = parseToolGrammarLeaks(text, ["dsml"]);
    expect(calls).toEqual([
      {
        grammar: "dsml",
        name: "bash",
        arguments: { command: "cd /tmp && ls", timeout: 600 },
      },
    ]);
  });

  it("parses opener-less DSML fragment with mangled <parameter name=\"tool\"> opener", () => {
    const text = `Let me check something.

<parameter name="bash">
<｜DSML｜ parameter name="command" string="true">cd /Users/mtolmacs/Projects/excalidraw && npx vitest run tests/bindingRepair.test.ts 2>&1 | head -40</｜DSML｜ parameter>
<｜DSML｜ parameter name="timeout" string="false">600</｜DSML｜ parameter>
</｜DSML｜ invoke>
</｜DSML｜ calls>`;

    const calls = parseToolGrammarLeaks(text, ["dsml"]);
    expect(calls).toEqual([
      {
        grammar: "dsml",
        name: "bash",
        arguments: {
          command: "cd /Users/mtolmacs/Projects/excalidraw && npx vitest run tests/bindingRepair.test.ts 2>&1 | head -40",
          timeout: 600,
        },
      },
    ]);
  });

  it("strips leaked spaced DSML markers from an unrecoverable fragment", () => {
    const message: MinimalAssistantMessage = {
      role: "assistant",
      content: [
        {
          type: "text",
          text: `Partial output.\n<｜DSML｜ parameter name="x">value</｜DSML｜ parameter>\n</｜DSML｜ invoke>`,
        },
      ],
      stopReason: "stop",
      timestamp: 1,
    };

    const result = repairAssistantMessageGrammarLeaks(message, enabledConfig, new Set(["bash"]));
    expect(result.changed).toBe(true);
    const text = (result.message.content[0] as { text: string }).text;
    expect(text).not.toContain("DSML");
  });
});

// DeepSeek V4.1 conflation: the invoke opener is lost entirely and the call
// collapses into a single prefix-less <parameter name="tool"> tag whose body
// carries the only argument, while the closer kept its DSML prefix:
//   <parameter name="bash">cd /tmp && ls</｜DSML｜ parameter>
describe("DSML conflated parameter calls (collapsed invoke)", () => {
  const dsmlMsg = (text: string): MinimalAssistantMessage => ({
    role: "assistant",
    content: [{ type: "text", text }],
    stopReason: "stop",
  });

  it("recovers a collapsed prefix-less parameter tag whose closer kept the DSML prefix", () => {
    const command = "cd /Users/mtolmacs/Projects/dexilion-team/imgproxy && sed -n '1,80p' imagedata/image_data_test.go";
    const text = `Checking the fixtures.\n\n<parameter name="bash">${command}</｜DSML｜ parameter>`;

    const result = repairAssistantMessageGrammarLeaks(dsmlMsg(text), enabledConfig, new Set(["bash"]));
    expect(result.changed).toBe(true);
    expect(result.recoveredCalls).toEqual([
      { grammar: "dsml", name: "bash", arguments: { command } },
    ]);
    expect(result.message.stopReason).toBe("toolUse");
    expect(result.message.content).toContainEqual(
      expect.objectContaining({ type: "toolCall", name: "bash", arguments: { command } }),
    );
    const stripped = (result.message.content[0] as { text: string }).text;
    expect(stripped).not.toContain("DSML");
    expect(stripped).not.toContain("<parameter");
  });

  it("recovers complete collapsed calls and strips a trailing truncated opener", () => {
    const text = [
      "<parameter name=\"bash\">cd /proj && ls testdata/test-images/jpg/ | head -30</｜DSML｜ parameter>",
      "",
      " <parameter name=\"bash\">cd /proj && ls testdata/test-images/png/</｜DSML｜ parameter>",
      "",
      " <parameter name=\"bash\">cd /proj && ls testdata/test-images/jpg/",
    ].join("\n");

    const result = repairAssistantMessageGrammarLeaks(dsmlMsg(text), enabledConfig, new Set(["bash"]));
    expect(result.recoveredCalls).toEqual([
      { grammar: "dsml", name: "bash", arguments: { command: "cd /proj && ls testdata/test-images/jpg/ | head -30" } },
      { grammar: "dsml", name: "bash", arguments: { command: "cd /proj && ls testdata/test-images/png/" } },
    ]);
    expect(result.message.stopReason).toBe("toolUse");
    const stripped = (result.message.content[0] as { text: string }).text;
    expect(stripped).not.toContain("DSML");
    expect(stripped).not.toContain("<parameter");
    // The truncated block's command stays as inert text, never executed.
    expect(stripped).toContain("ls testdata/test-images/jpg/");
  });

  it("maps a collapsed alias-named call onto bash", () => {
    const text = `<parameter name="command">git status --short</｜DSML｜ parameter>`;
    const result = repairAssistantMessageGrammarLeaks(dsmlMsg(text), enabledConfig, new Set(["bash"]));
    expect(result.recoveredCalls).toEqual([
      { grammar: "dsml", name: "bash", arguments: { command: "git status --short" } },
    ]);
  });

  it("recovers a bare-body DSML invoke body as the command for bash", () => {
    const text = `<｜DSML｜tool_calls>\n<｜DSML｜invoke name="bash">pwd</｜DSML｜invoke>\n</｜DSML｜tool_calls>`;
    const result = repairAssistantMessageGrammarLeaks(dsmlMsg(text), enabledConfig, new Set(["bash"]));
    expect(result.recoveredCalls).toEqual([
      { grammar: "dsml", name: "bash", arguments: { command: "pwd" } },
    ]);
    expect(result.message.stopReason).toBe("toolUse");
  });

  it("recovers the complete inner parameter of a truncated invoke fragment", () => {
    // Outer invoke/bash opener is truncated (no invoke closer) so it is only
    // stripped, but the inner name="command" child is itself complete and
    // self-describing: a conflated call with the command as its body.
    const text = `<parameter name="bash">\n<｜DSML｜ parameter name="command" string="true">cd /x && ls</｜DSML｜ parameter>`;

    const result = repairAssistantMessageGrammarLeaks(dsmlMsg(text), enabledConfig, new Set(["bash"]));
    expect(result.recoveredCalls).toEqual([
      { grammar: "dsml", name: "bash", arguments: { command: "cd /x && ls" } },
    ]);
    expect(result.message.stopReason).toBe("toolUse");
    const stripped = (result.message.content[0] as { text: string }).text;
    expect(stripped).not.toContain("DSML");
    expect(stripped).not.toContain("<parameter");
  });

  it("strips a truncated prefix-less parameter opener even without DSML evidence", () => {
    // deepinfra partially strips the DSML markers: the imgproxy session shows
    // truncated plain openers in messages with zero fullwidth bars anywhere.
    const text = "Let me look.\n<parameter name=\"bash\">cd /tmp && ls";
    const result = repairAssistantMessageGrammarLeaks(dsmlMsg(text), enabledConfig, new Set(["bash"]));
    expect(result.recoveredCalls).toEqual([]);
    expect(result.changed).toBe(true);
    const stripped = (result.message.content[0] as { text: string }).text;
    expect(stripped).not.toContain("<parameter");
    expect(stripped).toContain("cd /tmp && ls");
  });

  it("strips collapsed markers for non-command tools without recovering a call", () => {
    const text = `before\n<parameter name="read">/etc/hosts</｜DSML｜ parameter>\nafter`;
    const result = repairAssistantMessageGrammarLeaks(dsmlMsg(text), enabledConfig, new Set(["read", "bash"]));
    expect(result.recoveredCalls).toEqual([]);
    expect(result.changed).toBe(true);
    const stripped = (result.message.content[0] as { text: string }).text;
    expect(stripped).not.toContain("DSML");
    expect(stripped).not.toContain("<parameter");
    expect(stripped).toContain("before");
    expect(stripped).toContain("after");
  });

  // Shapes observed verbatim in the imgproxy session JSONL (deepinfra
  // deepseek-v4.1-flash): the provider partially strips the DSML markers, so
  // the same message stream can produce fully-plain, hybrid, or mixed debris.
  describe("partially de-markered debris (observed in session JSONL)", () => {
    it("recovers a fully-plain collapsed block with no DSML markers anywhere", () => {
      const text = `gofmt flags provider.go. Let me fix formatting.\n\n<parameter name="bash">cd /Users/mtolmacs/Projects/dexilion-team/imgproxy && gofmt -w info/provider.go</parameter>`;
      const result = repairAssistantMessageGrammarLeaks(dsmlMsg(text), enabledConfig, new Set(["bash"]));
      expect(result.recoveredCalls).toEqual([
        { grammar: "dsml", name: "bash", arguments: { command: "cd /Users/mtolmacs/Projects/dexilion-team/imgproxy && gofmt -w info/provider.go" } },
      ]);
      expect(result.message.stopReason).toBe("toolUse");
      expect((result.message.content[0] as { text: string }).text).not.toContain("<parameter");
    });

    it("recovers a plain collapsed call next to a native toolCall without duplicating it", () => {
      const message: MinimalAssistantMessage = {
        role: "assistant",
        content: [
          { type: "text", text: `Working.\n\n<parameter name="bash">cd /proj && gofmt -l .</parameter>` },
          { type: "toolCall", id: "native_1", name: "bash", arguments: { command: "cd /proj && gofmt -w ." } },
        ],
        stopReason: "toolUse",
      };
      const result = repairAssistantMessageGrammarLeaks(message, enabledConfig, new Set(["bash"]));
      expect(result.recoveredCalls).toEqual([]);
      expect(result.changed).toBe(true);
      expect((result.message.content[0] as { text: string }).text).not.toContain("<parameter");
      expect(result.message.content).toHaveLength(2);
    });

    it("strips an empty collapsed block without recovering it", () => {
      const text = `<parameter name="bash">\n</parameter>`;
      const result = repairAssistantMessageGrammarLeaks(dsmlMsg(text), enabledConfig, new Set(["bash"]));
      expect(result.recoveredCalls).toEqual([]);
      expect(result.changed).toBe(true);
      expect((result.message.content[0] as { text: string }).text.trim()).toBe("");
    });

    it("recovers a nested multi-parameter block closed with a hybrid tool-name closer", () => {
      // entry 204: <parameter name="read"> with plain children closed by </read>
      const text = [
        "I'll read the file.",
        "",
        "<parameter name=\"read\">",
        "<parameter name=\"path\">/Users/mtolmacs/Projects/dexilion-team/imgproxy/handlers/stream/handler_test.go</parameter>",
        "<parameter name=\"offset\">80</parameter>",
        "<parameter name=\"limit\">140</parameter>",
        "</read>",
      ].join("\n");
      const result = repairAssistantMessageGrammarLeaks(dsmlMsg(text), enabledConfig, new Set(["read"]));
      expect(result.recoveredCalls).toEqual([
        {
          grammar: "dsml",
          name: "read",
          arguments: {
            path: "/Users/mtolmacs/Projects/dexilion-team/imgproxy/handlers/stream/handler_test.go",
            offset: 80,
            limit: 140,
          },
        },
      ]);
      expect(result.message.stopReason).toBe("toolUse");
    });

    it("parses mixed barred/plain children into full arguments (garbage read regression)", () => {
      // The old parser recovered read {"limit": "140"} with no path from this
      // mixed-debris shape; all children must land in the argument object.
      const text = [
        "<parameter name=\"read\">",
        "<parameter name=\"path\">/proj/handler_test.go</parameter>",
        "<parameter name=\"offset\">80</parameter>",
        "<｜DSML｜ parameter name=\"limit\" string=\"true\">140</｜DSML｜ parameter>",
        "</｜DSML｜ invoke>",
      ].join("\n");
      const calls = parseToolGrammarLeaks(text, ["dsml"]);
      expect(calls).toEqual([
        {
          grammar: "dsml",
          name: "read",
          arguments: { path: "/proj/handler_test.go", offset: 80, limit: "140" },
        },
      ]);
    });

    it("does not duplicate a recovered call when two parsers produce the same range", () => {
      const text = [
        "<parameter name=\"read\">",
        "<｜DSML｜ parameter name=\"path\" string=\"true\">/proj/a.go</｜DSML｜ parameter>",
        "</｜DSML｜ invoke>",
      ].join("\n");
      const result = repairAssistantMessageGrammarLeaks(dsmlMsg(text), enabledConfig, new Set(["read"]));
      expect(result.recoveredCalls).toEqual([
        { grammar: "dsml", name: "read", arguments: { path: "/proj/a.go" } },
      ]);
    });
  });
});

describe("assistant message grammar repair", () => {
  it("strips leaked text and appends a recovered toolCall", () => {
    const message: MinimalAssistantMessage = {
      role: "assistant",
      content: [
        {
          type: "text",
          text: `I'll use the tool.
<｜DSML｜tool_calls>
<｜DSML｜invoke name="bash">
<｜DSML｜parameter name="command" string="true">pwd</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>`,
        },
      ],
      stopReason: "stop",
      timestamp: 1,
    };

    const result = repairAssistantMessageGrammarLeaks(message, enabledConfig, new Set(["bash"]));

    expect(result.changed).toBe(true);
    expect(result.message.stopReason).toBe("toolUse");
    expect(result.message.content).toEqual([
      { type: "text", text: "I'll use the tool." },
      {
        type: "toolCall",
        id: expect.stringMatching(/^tool_repair_dsml_/),
        name: "bash",
        arguments: { command: "pwd" },
      },
    ]);
  });

  it("does not recover unknown tools, but strips their leaked markers", () => {
    const message: MinimalAssistantMessage = {
      role: "assistant",
      content: [
        { type: "text", text: `<tool_call>{"name":"unknown","arguments":{"path":"/foo"}}</tool_call>` },
      ],
      stopReason: "stop",
      timestamp: 1,
    };

    const result = repairAssistantMessageGrammarLeaks(message, enabledConfig, new Set(["bash"]));
    expect(result.recoveredCalls).toEqual([]);
    expect(result.message.stopReason).toBe("stop");
    expect((result.message.content[0] as { text: string }).text).not.toContain("tool_call");
  });

  it("does not recover a call with empty arguments", () => {
    // GLM-style empty tool calls (e.g. `<tool_call>write</tool_call>`) would
    // otherwise be promoted to a native `toolCall` block with `{}` arguments,
    // causing a validation error when pi tries to execute them. They should be
    // stripped from the text but not recovered.
    const message: MinimalAssistantMessage = {
      role: "assistant",
      content: [{ type: "text", text: `<tool_call>write</tool_call>` }],
      stopReason: "stop",
      timestamp: 1,
    };

    const result = repairAssistantMessageGrammarLeaks(message, enabledConfig, new Set(["write"]));
    expect(result.recoveredCalls).toEqual([]);
    expect(result.message.stopReason).toBe("stop");
    expect((result.message.content[0] as { text: string }).text).not.toContain("tool_call");
  });

  it("strips a truncated DSML open marker (stream died mid-token, issue #3712)", () => {
    const message: MinimalAssistantMessage = {
      role: "assistant",
      content: [{ type: "text", text: "I'll read the file.\n<｜DSML｜tool_calls" }],
      stopReason: "stop",
      timestamp: 1,
    };

    const result = repairAssistantMessageGrammarLeaks(message, enabledConfig, new Set(["read"]));
    expect(result.changed).toBe(true);
    expect(result.recoveredCalls).toEqual([]);
    expect(result.message.stopReason).toBe("stop");
    expect(result.message.content).toEqual([{ type: "text", text: "I'll read the file." }]);
  });

  it("strips orphan DSML markers from a truncated body without recovering a call", () => {
    const message: MinimalAssistantMessage = {
      role: "assistant",
      content: [{
        type: "text",
        text: "I'll inspect.\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"read\">\n<｜DSML｜parameter name=\"path\" string=\"true\">/foo",
      }],
      stopReason: "stop",
      timestamp: 1,
    };

    const result = repairAssistantMessageGrammarLeaks(message, enabledConfig, new Set(["read"]));
    expect(result.changed).toBe(true);
    expect(result.recoveredCalls).toEqual([]);
    expect(result.message.stopReason).toBe("stop");
    const text = (result.message.content[0] as { text: string }).text;
    expect(text).not.toContain("DSML");
    expect(text).toContain("I'll inspect.");
    expect(text).toContain("/foo");
  });

  it("strips dangling DSML markers in strip mode too", () => {
    const stripConfig: GrammarRepairConfig = { ...enabledConfig, mode: "strip" };
    const message: MinimalAssistantMessage = {
      role: "assistant",
      content: [{ type: "text", text: "hi\n<｜DSML｜tool_calls" }],
      stopReason: "stop",
      timestamp: 1,
    };

    const result = repairAssistantMessageGrammarLeaks(message, stripConfig, new Set());
    expect(result.changed).toBe(true);
    expect((result.message.content[0] as { text: string }).text).toBe("hi");
  });

  it("does not strip a truncated DSML marker inside a code fence", () => {
    const message: MinimalAssistantMessage = {
      role: "assistant",
      content: [{ type: "text", text: "```\n<｜DSML｜tool_calls\n```" }],
      stopReason: "stop",
      timestamp: 1,
    };

    const result = repairAssistantMessageGrammarLeaks(message, enabledConfig, new Set(["read"]));
    expect(result.changed).toBe(false);
  });

  it("does not double-strip dangling markers already covered by a complete DSML block", () => {
    const message: MinimalAssistantMessage = {
      role: "assistant",
      content: [{
        type: "text",
        text: `prefix
<｜DSML｜tool_calls>
<｜DSML｜invoke name="bash">
<｜DSML｜parameter name="command" string="true">pwd</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>`,
      }],
      stopReason: "stop",
      timestamp: 1,
    };

    const result = repairAssistantMessageGrammarLeaks(message, enabledConfig, new Set(["bash"]));
    expect(result.recoveredCalls).toHaveLength(1);
    expect(result.recoveredCalls[0]).toEqual({ grammar: "dsml", name: "bash", arguments: { command: "pwd" } });
    expect((result.message.content[0] as { text: string }).text).toBe("prefix");
  });
});

describe("per-model grammar repair enablement", () => {
  const baseConfig: GrammarRepairConfig = {
    enabled: false,
    grammars: ["qwen"],
    mode: "recover",
    requireKnownTool: true,
    debug: false,
    leakModels: ["qwen3", "^@cf/.+kimi"],
  };

  const leakedMessage = (): MinimalAssistantMessage => ({
    role: "assistant",
    content: [{
      type: "text",
      text: `<tool_call>
{"name": "bash", "arguments": {"command": "pwd"}}
</tool_call>`,
    }],
    stopReason: "stop",
    timestamp: 1,
  });

  it("keeps grammar repair off when no leakModels are configured", () => {
    const { leakModels: _omitted, ...config } = baseConfig;
    expect(resolveGrammarRepairForModel(config, { id: "local/qwen3-coder" }).enabled).toBe(false);
  });

  it("enables recovery when the model id matches a leakModels pattern", () => {
    const resolved = resolveGrammarRepairForModel(baseConfig, { id: "local/qwen3-coder" });
    expect(resolved.enabled).toBe(true);
  });

  it("matches patterns case-insensitively", () => {
    expect(resolveGrammarRepairForModel(baseConfig, { id: "LOCAL/QWEN3-CODER" }).enabled).toBe(true);
  });

  it("leaves recovery off for models that match no pattern", () => {
    expect(resolveGrammarRepairForModel(baseConfig, { id: "claude-sonnet-4" }).enabled).toBe(false);
  });

  it("leaves recovery off when there is no model id", () => {
    expect(resolveGrammarRepairForModel(baseConfig, undefined).enabled).toBe(false);
    expect(resolveGrammarRepairForModel(baseConfig, {}).enabled).toBe(false);
  });

  it("returns the config unchanged when repair is already globally enabled", () => {
    const globalConfig = { ...baseConfig, enabled: true };
    expect(resolveGrammarRepairForModel(globalConfig, { id: "claude-sonnet-4" })).toBe(globalConfig);
  });

  it("normalization keeps compilable regex strings and drops invalid ones", () => {
    const normalized = normalizeGrammarRepairConfig({
      leakModels: ["qwen3", "[", 1, null] as unknown as string[],
    });
    expect(normalized.leakModels).toEqual(["qwen3"]);
    expect(normalizeGrammarRepairConfig({}).leakModels).toBeUndefined();
  });

  it("writes the recovered tool call back onto a matched bleeding message", () => {
    const resolved = resolveGrammarRepairForModel(baseConfig, { id: "local/qwen3-coder" });
    const result = repairAssistantMessageGrammarLeaks(leakedMessage(), resolved, new Set(["bash"]));
    expect(result.changed).toBe(true);
    expect(result.recoveredCalls).toEqual([
      { grammar: "qwen", name: "bash", arguments: { command: "pwd" } },
    ]);
    expect(result.message.stopReason).toBe("toolUse");
    expect(result.message.content).toContainEqual(
      expect.objectContaining({ type: "toolCall", name: "bash", arguments: { command: "pwd" } }),
    );
    expect((result.message.content[0] as { text: string }).text.trim()).toBe("");
  });

  it("leaves bleeding text intact for a non-matching model", () => {
    const resolved = resolveGrammarRepairForModel(baseConfig, { id: "claude-sonnet-4" });
    const result = repairAssistantMessageGrammarLeaks(leakedMessage(), resolved, new Set(["bash"]));
    expect(result.changed).toBe(false);
    expect(result.recoveredCalls).toHaveLength(0);
  });
});

describe("DSML tool-name aliasing", () => {
  const P = "｜｜DSML｜｜";
  const dsml = (body: string) => `<${P}tool_calls>\n${body}\n</${P}tool_calls>`;
  const dsmlMsg = (text: string): MinimalAssistantMessage => ({
    role: "assistant",
    content: [{ type: "text", text }],
    stopReason: "stop",
  });

  it(`rewrites invoke name="command" with a command parameter to bash`, () => {
    const text = dsml(`<${P}invoke name="command">\n<${P}parameter name="command" string="true">pwd</${P}parameter>\n</${P}invoke>`);
    const result = repairAssistantMessageGrammarLeaks(dsmlMsg(text), enabledConfig, new Set(["bash", "read"]));
    expect(result.changed).toBe(true);
    expect(result.recoveredCalls).toEqual([{ grammar: "dsml", name: "bash", arguments: { command: "pwd" } }]);
    expect(result.message.stopReason).toBe("toolUse");
    expect(result.message.content).toContainEqual(
      expect.objectContaining({ type: "toolCall", name: "bash", arguments: { command: "pwd" } }),
    );
  });

  it(`recovers the bare invoke body of name="command" as the command`, () => {
    const text = dsml(`<${P}invoke name="command">pwd</${P}invoke>`);
    const result = repairAssistantMessageGrammarLeaks(dsmlMsg(text), enabledConfig, new Set(["bash"]));
    expect(result.recoveredCalls).toEqual([{ grammar: "dsml", name: "bash", arguments: { command: "pwd" } }]);
    expect(result.message.stopReason).toBe("toolUse");
  });

  it(`maps cmd/shell aliases onto bash with a command argument`, () => {
    const cmdArg = dsml(`<${P}invoke name="command">\n<${P}parameter name="cmd" string="true">ls -la</${P}parameter>\n</${P}invoke>`);
    expect(repairAssistantMessageGrammarLeaks(dsmlMsg(cmdArg), enabledConfig, new Set(["bash"])).recoveredCalls)
      .toEqual([{ grammar: "dsml", name: "bash", arguments: { command: "ls -la" } }]);

    const shell = dsml(`<${P}invoke name="shell">echo hi</${P}invoke>`);
    expect(repairAssistantMessageGrammarLeaks(dsmlMsg(shell), enabledConfig, new Set(["bash"])).recoveredCalls)
      .toEqual([{ grammar: "dsml", name: "bash", arguments: { command: "echo hi" } }]);
  });

  it(`leaves invoke name="bash" untouched`, () => {
    const text = dsml(`<${P}invoke name="bash">\n<${P}parameter name="command" string="true">pwd</${P}parameter>\n</${P}invoke>`);
    const result = repairAssistantMessageGrammarLeaks(dsmlMsg(text), enabledConfig, new Set(["bash"]));
    expect(result.recoveredCalls).toEqual([{ grammar: "dsml", name: "bash", arguments: { command: "pwd" } }]);
  });

  it(`strips the raw markers when the alias target is not active`, () => {
    const text = dsml(`<${P}invoke name="command">pwd</${P}invoke>`);
    const result = repairAssistantMessageGrammarLeaks(dsmlMsg(text), enabledConfig, new Set(["read"]));
    expect(result.recoveredCalls).toEqual([]);
    expect(result.message.stopReason).toBe("stop");
    expect((result.message.content[0] as { text: string }).text).not.toContain("DSML");
  });

  it(`strips an unaliased unknown tool leak without recovering it`, () => {
    const text = dsml(`<${P}invoke name="totally_unknown">\n<${P}parameter name="x" string="true">1</${P}parameter>\n</${P}invoke>`);
    const result = repairAssistantMessageGrammarLeaks(dsmlMsg(`checking\n${text}`), enabledConfig, new Set(["bash"]));
    expect(result.recoveredCalls).toEqual([]);
    expect(result.message.stopReason).toBe("stop");
    const stripped = (result.message.content[0] as { text: string }).text;
    expect(stripped).not.toContain("DSML");
    expect(stripped).toContain("checking");
  });

  it(`honors a configured toolNameAliases override`, () => {
    const config: GrammarRepairConfig = { ...enabledConfig, toolNameAliases: { command: "read" } };
    const text = dsml(`<${P}invoke name="command">\n<${P}parameter name="command" string="true">pwd</${P}parameter>\n</${P}invoke>`);
    const result = repairAssistantMessageGrammarLeaks(dsmlMsg(text), config, new Set(["bash", "read"]));
    expect(result.recoveredCalls).toEqual([{ grammar: "dsml", name: "read", arguments: { command: "pwd" } }]);
  });
});
