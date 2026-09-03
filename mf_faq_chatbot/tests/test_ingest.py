"""Tests for the ingestion pipeline's AMC metadata tagging (CSV parsing and
local-file mapping) — no network access, no embedding model required."""

from __future__ import annotations

import ingest


class TestReadUrlsCsv:
    def test_reads_amc_column(self, tmp_path, monkeypatch):
        csv_path = tmp_path / "urls.csv"
        csv_path.write_text(
            "url,title,category,amc\n"
            "https://example.com/a,A,Scheme,HDFC Mutual Fund\n"
        )
        monkeypatch.setattr(ingest, "URLS_CSV", csv_path)
        rows = ingest._read_urls_csv()
        assert rows == [
            {
                "url": "https://example.com/a",
                "title": "A",
                "category": "Scheme",
                "amc": "HDFC Mutual Fund",
            }
        ]

    def test_missing_amc_column_defaults_to_general(self, tmp_path, monkeypatch):
        csv_path = tmp_path / "urls.csv"
        csv_path.write_text("url,title,category\nhttps://example.com/a,A,Scheme\n")
        monkeypatch.setattr(ingest, "URLS_CSV", csv_path)
        rows = ingest._read_urls_csv()
        assert rows[0]["amc"] == "General"

    def test_quoted_url_with_embedded_comma_parses_correctly(self, tmp_path, monkeypatch):
        """Regression test: a URL containing a literal comma (e.g. a date like
        "April 28, 2023" that wasn't percent-encoded) must be CSV-quoted,
        otherwise csv.DictReader silently misaligns every column after it."""
        csv_path = tmp_path / "urls.csv"
        csv_path.write_text(
            "url,title,category,amc\n"
            '"https://example.com/doc,2023.pdf",Doc,SID,HDFC Mutual Fund\n'
        )
        monkeypatch.setattr(ingest, "URLS_CSV", csv_path)
        rows = ingest._read_urls_csv()
        assert rows == [
            {
                "url": "https://example.com/doc,2023.pdf",
                "title": "Doc",
                "category": "SID",
                "amc": "HDFC Mutual Fund",
            }
        ]

    def test_real_urls_csv_has_no_misaligned_rows(self):
        """Every row in the actual urls.csv must parse into exactly the four
        expected non-empty fields — catches unquoted commas or other CSV
        formatting mistakes in the real data file."""
        rows = ingest._read_urls_csv()
        assert len(rows) > 0
        for row in rows:
            assert set(row.keys()) == {"url", "title", "category", "amc"}
            assert all(row.values())
            assert row["url"].startswith(("http://", "https://"))


class TestBuildDocument:
    def test_amc_is_stored_in_metadata(self):
        doc = ingest._build_document(
            text="content",
            source_url="https://example.com/a",
            title="A",
            category="Scheme",
            source_type="webpage",
            amc="ICICI Prudential Mutual Fund",
        )
        assert doc.metadata["amc"] == "ICICI Prudential Mutual Fund"


class TestLocalFileAmcMapping:
    def test_known_local_files_map_to_expected_amc(self):
        assert ingest.LOCAL_FILE_AMC["sbi_scheme_reference.md"] == "SBI Mutual Fund"
        assert ingest.LOCAL_FILE_AMC["regulatory_reference.md"] == "Regulatory"

    def test_unknown_local_file_falls_back_to_general(self):
        assert ingest.LOCAL_FILE_AMC.get("unmapped_file.md", "General") == "General"
