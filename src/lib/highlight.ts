import { codeToHtml } from "shiki";

export async function highlightCode(
  code: string,
  lang: string = "python",
  title?: string,
): Promise<string> {
  return codeToHtml(code.trimEnd(), {
    lang,
    theme: "github-dark-default",
    transformers: [
      {
        pre(node) {
          node.properties.tabindex = "0";
          node.properties.role = "region";
          node.properties["aria-label"] = title ? `Code snippet: ${title}` : "Code snippet";
        },
      },
    ],
  });
}
