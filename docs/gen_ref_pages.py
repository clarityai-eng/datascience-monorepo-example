#!/usr/bin/env python
"""Generate the code reference pages."""

from pathlib import Path

import mkdocs_gen_files


def gen_ref_pages():
    for path in sorted(Path("src").rglob("*.py")):  #
        module_path = path.relative_to("src").with_suffix("")  #
        doc_path = path.relative_to("src").with_suffix(".md")  #
        full_doc_path = Path("reference", doc_path)  #

        parts = list(module_path.parts)

        if parts[-1] == "__main__" or parts[-1] == "__init__":
            continue

        with mkdocs_gen_files.open(full_doc_path, "w") as fd:  #
            identifier = ".".join(parts)  #
            print("::: " + identifier, file=fd)  #

        mkdocs_gen_files.set_edit_path(full_doc_path, path)  #


def add_index_page_from_readme():
    readme = Path("README.md").read_text()
    with mkdocs_gen_files.open("index.md", "w") as f:
        f.write(readme)

    mkdocs_gen_files.set_edit_path("index.md", "README.md")  #


gen_ref_pages()
add_index_page_from_readme()
