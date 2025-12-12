from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_txt_directory(source_dir: Path, output_dir: Path, chunk_size: int, chunk_overlap: int) -> None:
    """Split all txt files in a directory into chunks with specified size and overlap.

    Args:
        source_dir (Path): Path to the source directory.
        output_dir (Path): Path to the output directory.
        chunk_size (int): Maximum number of characters per chunk, must be greater than 0.
        chunk_overlap (int): Number of overlapping characters between chunks, must be between 0 and chunk_size.

    Raises
    ------
    ValueError
        If source is not a directory or chunk parameters are invalid.
    """
    if not source_dir.is_dir():
        error_message = "source_dir must be a directory path."
        raise ValueError(error_message)
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        error_message = "chunk_size must be > 0, and chunk_overlap must be less than chunk_size."
        raise ValueError(error_message)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    for file_path in source_dir.rglob("*.txt"):
        content = file_path.read_text(encoding="utf-8")
        chunks = text_splitter.split_text(content)

        dest_dir = output_dir / file_path.relative_to(source_dir).parent
        dest_dir.mkdir(parents=True, exist_ok=True)

        for index, chunk in enumerate(chunks, start=1):
            dest_file = dest_dir / f"{file_path.stem}_chunk_{index:04d}.txt"
            dest_file.write_text(chunk, encoding="utf-8")
