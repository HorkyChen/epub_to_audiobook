import logging
import os
from ebooklib import epub
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def extract_toc_items(toc):
    """
    递归提取 TOC 中的所有条目，返回列表，每个元素为 (标题, href)
    无论 TOC 结构是 Section/Link 还是 tuple，均统一提取
    """
    items = []

    def _extract(entries):
        for entry in entries:
            if isinstance(entry, epub.Link):
                items.append((entry.title, entry.href))
            elif isinstance(entry, epub.Section):
                items.append((entry.title, entry.href))
                if entry.children:
                    _extract(entry.children)
            elif isinstance(entry, tuple) and len(entry) == 2:
                section, children = entry
                if hasattr(section, "title") and hasattr(section, "href"):
                    items.append((section.title, section.href))
                else:
                    items.append((str(section), None))
                if children:
                    _extract(children)
            else:
                logging.warning(f"未知的 TOC 条目类型: {entry}")

    _extract(toc)
    return items


def get_html_content(book, href):
    """
    根据 href 从 EPUB 中获取对应的 HTML 文件内容，并解码为字符串
    """
    # 提取文件名（去掉锚点部分）
    file_name = href.split("#")[0]
    # 通过 href 获取对应的 item
    item = book.get_item_with_href(file_name)
    if item is None:
        # 某些 EPUB 中 href 可能不包含文件名（理论上不会），尝试直接找
        for it in book.get_items():
            if it.get_name().endswith(file_name):
                item = it
                break
    if item is None:
        raise ValueError(f"未找到文件: {file_name}")
    # 获取内容并解码（假设 UTF-8 编码）
    content = item.get_content()
    return content.decode("utf-8")


def extract_chapters(epub_path):
    """
    主函数：读取 EPUB，提取 TOC 并按锚点分割章节内容
    """
    # 1. 读取 EPUB
    book = epub.read_epub(epub_path)
    toc = book.toc

    # 2. 提取所有 TOC 条目
    toc_items = extract_toc_items(toc)
    logging.info(f"共发现 {len(toc_items)} 个 TOC 条目")

    # 3. 由于所有条目都指向同一个 HTML 文件，我们只需读取一次
    if not toc_items:
        return

    # 取出第一个条目的文件名（所有条目应相同，取第一个即可）
    first_href = toc_items[0][1]
    base_file = first_href.split("#")[0]
    html_str = get_html_content(book, first_href)
    soup = BeautifulSoup(html_str, "html.parser")

    from bs4 import NavigableString

    # 4. 收集每个标题及其对应的有效锚点元素
    # 我们将有效的 (title, elem) 放在一个列表中
    chapters = []
    for title, href in toc_items:
        frag = href.split("#")[-1] if "#" in href else None
        if not frag:
            continue

        elems = soup.select(f'[id="{frag}"], [name="{frag}"]')
        if elems:
            chapters.append((title, elems[0]))

    # 5. 按顺序遍历所有的有效锚点，分割内容
    chapters_text = []
    for i, (title, current_elem) in enumerate(chapters):
        # 如果不是最后一个，它的终点就是下一个锚点元素
        next_anchor = chapters[i + 1][1] if i + 1 < len(chapters) else None

        # 收集文本：遍历 current_elem.next_elements 并在遇到 next_anchor 时停止
        text_parts = []
        for elem in current_elem.next_elements:
            if elem == next_anchor:
                break
            if isinstance(elem, NavigableString):
                text_parts.append(str(elem))

        full_text = "".join(text_parts).strip()
        chapters_text.append((title, full_text))

    return chapters_text


if __name__ == "__main__":
    # check the arguments to get the input file
    import argparse
    config = argparse.ArgumentParser(description="Test EpubBookParser")
    config.add_argument("input_file", help="Path to the EPUB file to test")
    args = config.parse_args()

    if not os.path.exists(args.input_file):
        logging.error(f"Input file does not exist: {args.input_file}")
        exit(1)
    logging.info(f"Testing EpubBookParser with file: {args.input_file}")
    epub_file = args.input_file
    chapters = extract_chapters(epub_file)
    for title, text in chapters:
        print(f"\n=== {title} ===")
        print(text[:500])  # 只打印前500字符预览
