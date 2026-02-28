filepath = "agent_gantry/core/router.py"
with open(filepath) as f:
    content = f.read()


search_block = """        # Use cached embeddings if available, otherwise re-embed (fallback)
        if cached_embeddings:
            embeddings = []
            for tool, _ in scored_tools:
                tool_key = f"{tool.namespace}.{tool.name}"
                embedding = cached_embeddings.get(tool_key)
                if embedding is None:
                    # Fallback: re-embed this specific tool if not in cache
                    embedding = await self._embedder.embed_text(tool.to_searchable_text())
                embeddings.append(embedding)
        else:
            # Fallback: re-embed all tools (old behavior for backward compatibility)
            tool_texts = [tool.to_searchable_text() for tool, _ in scored_tools]
            embeddings = await self._embedder.embed_batch(tool_texts)"""

replace_block = """        # Use cached embeddings if available, otherwise re-embed (fallback)
        if cached_embeddings:
            embeddings = []
            missing_tools_indices = []
            missing_tool_texts = []

            for idx, (tool, _) in enumerate(scored_tools):
                tool_key = f"{tool.namespace}.{tool.name}"
                embedding = cached_embeddings.get(tool_key)
                if embedding is None:
                    missing_tools_indices.append(idx)
                    missing_tool_texts.append(tool.to_searchable_text())
                    embeddings.append([])  # Placeholder
                else:
                    embeddings.append(embedding)

            if missing_tool_texts:
                # Fallback: batch embed missing tools
                missing_embeddings = await self._embedder.embed_batch(missing_tool_texts)
                for idx, emb in zip(missing_tools_indices, missing_embeddings):
                    embeddings[idx] = emb
        else:
            # Fallback: re-embed all tools (old behavior for backward compatibility)
            tool_texts = [tool.to_searchable_text() for tool, _ in scored_tools]
            embeddings = await self._embedder.embed_batch(tool_texts)"""

new_content = content.replace(search_block, replace_block)
if new_content != content:
    with open(filepath, "w") as f:
        f.write(new_content)
    print("Patched successfully")
else:
    print("Search block not found")
