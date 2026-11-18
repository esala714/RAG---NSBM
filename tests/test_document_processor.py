from pathlib import Path

from src.ingestion.document_processor import (
    clean_text,
    split_into_chunks,
    infer_category,
    process_file
)


# Test 1: Text Cleaning
def test_clean_text():
    text = "  Hello\x00   World \t Python  "

    result = clean_text(text)

    assert result == "Hello World Python"


# Test 2: Chunk Overlap
def test_chunk_overlap():
    text = "one two three four five six seven eight"

    chunks = split_into_chunks(
        text,
        chunk_size=4,
        overlap=2
    )

    assert len(chunks) == 3
    assert chunks[0] == "one two three four"
    assert chunks[1] == "three four five six"
    assert chunks[2] == "five six seven eight"


# Test 2b: Default chunks fit what MiniLM can read (about 192 words)
def test_default_chunks_fit_minilm():
    words = [f"w{i}" for i in range(1000)]

    chunks = split_into_chunks(" ".join(words))

    assert all(len(c.split()) <= 180 for c in chunks)
    assert chunks[1].split()[0] == "w140"   # 180 - 40 overlap: the next chunk starts 140 words on


# Test 3: Empty Text
def test_empty_text():
    result = split_into_chunks("")

    assert result == []


# Test 4: Category Detection
def test_infer_category():
    result = infer_category(
        Path("NSBM Academic Calendar.pdf")
    )

    assert result == "academic_calendar"


# Test 4b: Lecturer documents get their own category
def test_infer_category_staff():
    result = infer_category(
        Path("NSBM_Lecturers_Qualifications_Updated_2026.pdf")
    )

    assert result == "staff"


# Test 5: Metadata Extraction
def test_process_file_metadata(tmp_path):

    file_path = tmp_path / "NSBM Attendance Policy.txt"

    file_path.write_text(
        "Students must attend lectures and tutorials.",
        encoding="utf-8"
    )

    records = process_file(file_path)

    assert len(records) == 1

    record = records[0]
    metadata = record["metadata"]

    assert record["text"] == (
        "Students must attend lectures and tutorials."
    )

    assert metadata["source"] == "NSBM Attendance Policy.txt"
    assert metadata["category"] == "attendance"
    assert metadata["file_type"] == "txt"
    assert metadata["page_number"] is None
    assert metadata["chunk_index"] == 1
    assert metadata["word_count"] == 6
    assert len(metadata["doc_hash"]) == 64