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
        assert ingest.LOCAL_FILE_AMC["kuvera_platform_guide.md"] == "Platform"
        assert ingest.LOCAL_FILE_AMC["regulatory_reference.md"] == "Regulatory"

    def test_unknown_local_file_falls_back_to_general(self):
        assert ingest.LOCAL_FILE_AMC.get("unmapped_file.md", "General") == "General"
