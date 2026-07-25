from __future__ import annotations

import io
import json
import threading
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from umux_processor.api import ApiSettings, create_app
from umux_processor.pipeline import PipelineServiceError


HEADER = (
    "response_id,submitted_at,product,product_version,platform,country,"
    "user_segment,score1,score2\n"
)
ROW = "response-1,2024-01-02 03:04:05,Payments,2.0,Web,US,New,5,4\n"
ARTIFACTS = {
    "cleaned_responses.csv",
    "rejected_responses.csv",
    "monthly_aggregates.csv",
    "product_summary.csv",
    "quality_summary.json",
    "dashboard.html",
}


def app(tmp_path: Path, **overrides: int) -> object:
    return create_app(ApiSettings(
        temp_root=tmp_path / "requests",
        max_files=overrides.get("max_files", 3),
        max_file_bytes=overrides.get("max_file_bytes", 10_000),
        max_request_bytes=overrides.get("max_request_bytes", 20_000),
    ))


def csv_file(name: str = "responses.csv", contents: str = HEADER + ROW) -> tuple[str, io.BytesIO, str]:
    return name, io.BytesIO(contents.encode("utf-8")), "text/csv"


def artifact_names(response: object) -> set[str]:
    with zipfile.ZipFile(io.BytesIO(response.content)) as bundle:  # type: ignore[attr-defined]
        return set(bundle.namelist())


def test_health_endpoint_reports_ready(tmp_path: Path) -> None:
    with TestClient(app(tmp_path)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_processes_a_single_file_through_the_shared_pipeline_service(tmp_path: Path, monkeypatch: object) -> None:
    called = False
    from umux_processor import api
    actual_service = api.run_pipeline_service

    def record_call(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        return actual_service(*args, **kwargs)

    monkeypatch.setattr(api, "run_pipeline_service", record_call)  # type: ignore[attr-defined]
    with TestClient(app(tmp_path)) as client:
        response = client.post("/process", files=[("files", csv_file())])

    assert response.status_code == 200
    assert called
    assert response.headers["content-type"].startswith("application/zip")
    assert artifact_names(response) == ARTIFACTS


def test_processes_multiple_files_and_keeps_partial_rejections(tmp_path: Path) -> None:
    first = HEADER + ROW
    second = HEADER + "bad,2024-01-03 03:04:05,Payments,2.0,Web,US,New,bad,4\n"
    with TestClient(app(tmp_path)) as client:
        response = client.post("/process", files=[
            ("files", csv_file("first.csv", first)),
            ("files", csv_file("second.csv", second)),
        ])

    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as bundle:
        assert len(bundle.read("cleaned_responses.csv").splitlines()) == 2
        assert len(bundle.read("rejected_responses.csv").splitlines()) == 2


def test_maps_schema_and_malformed_csv_to_safe_client_errors(tmp_path: Path) -> None:
    with TestClient(app(tmp_path)) as client:
        schema = client.post("/process", files=[("files", csv_file("invalid.csv", "response_id\nonly\n"))])
        malformed = client.post("/process", files=[("files", csv_file("broken.csv", HEADER + ROW.rstrip("\n") + ",extra\n"))])

    for response in (schema, malformed):
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "INPUT_ERROR"
        assert "C:\\" not in response.text


def test_rejects_unsupported_media_types_and_filename_traversal(tmp_path: Path) -> None:
    with TestClient(app(tmp_path)) as client:
        unsupported = client.post("/process", files={"files": ("data.txt", HEADER + ROW, "text/plain")})
        traversal = client.post("/process", files=[("files", csv_file("../../outside.csv"))])
        malformed_multipart = client.post(
            "/process",
            content=b"--not-the-declared-boundary\r\n",
            headers={"content-type": "multipart/form-data; boundary=declared-boundary"},
        )

    assert unsupported.status_code == 415
    assert unsupported.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"
    assert traversal.status_code == 200
    assert malformed_multipart.status_code == 400
    assert malformed_multipart.json()["error"]["code"] == "MALFORMED_MULTIPART"
    with zipfile.ZipFile(io.BytesIO(traversal.content)) as bundle:
        assert ".." not in bundle.read("cleaned_responses.csv").decode("utf-8")


def test_enforces_file_count_and_size_limits(tmp_path: Path) -> None:
    with TestClient(app(tmp_path, max_files=1, max_file_bytes=100, max_request_bytes=1_000)) as client:
        too_many = client.post("/process", files=[("files", csv_file("one.csv")), ("files", csv_file("two.csv"))])
        too_large = client.post("/process", files=[("files", csv_file(contents=HEADER + "x" * 200))])
    with TestClient(app(tmp_path, max_files=3, max_file_bytes=10_000, max_request_bytes=300)) as client:
        total_too_large = client.post("/process", files=[("files", csv_file(contents=HEADER + ROW + "x" * 200))])

    assert too_many.status_code == 413
    assert too_many.json()["error"]["code"] == "TOO_MANY_FILES"
    assert too_large.status_code == 413
    assert too_large.json()["error"]["code"] == "FILE_TOO_LARGE"
    assert total_too_large.status_code == 413
    assert total_too_large.json()["error"]["code"] == "REQUEST_TOO_LARGE"


def test_accepts_uploaded_toml_configuration_but_rejects_invalid_configuration(tmp_path: Path) -> None:
    valid_config = (
        "[products.aliases]\npayment = 'Payments'\n\n"
        "[platforms]\nsupported = ['Web']\n\n"
        "[user_segments.aliases]\nnew = 'New'\n\n"
        "[countries]\ncase = 'upper'\nmissing_value = 'Unknown'\ncode_length = 2\n\n"
        "[timestamps]\naccepted_format = '%Y-%m-%d %H:%M:%S'\n\n"
        "[report]\nsmall_sample_threshold = 5\n"
    )
    with TestClient(app(tmp_path)) as client:
        configured = client.post("/process", files=[
            ("files", csv_file(contents=HEADER + ROW.replace("Payments", "Payment"))),
            ("config", ("normalization.toml", valid_config, "application/toml")),
        ])
        invalid = client.post("/process", files=[
            ("files", csv_file()),
            ("config", ("normalization.toml", "not = [valid", "application/toml")),
        ])

    assert configured.status_code == 200
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "INPUT_ERROR"


def test_reconciles_zip_contents_and_removes_request_temporary_files(tmp_path: Path) -> None:
    application = app(tmp_path)
    with TestClient(application) as client:
        response = client.post("/process", files=[("files", csv_file())])

    with zipfile.ZipFile(io.BytesIO(response.content)) as bundle:
        quality = json.loads(bundle.read("quality_summary.json"))
        assert len(bundle.read("cleaned_responses.csv").splitlines()) - 1 == quality["accepted_row_count"]
        assert len(bundle.read("rejected_responses.csv").splitlines()) - 1 == quality["rejected_row_count"]
    assert list((tmp_path / "requests").iterdir()) == []


def test_concurrent_requests_have_independent_artifacts(tmp_path: Path) -> None:
    application = app(tmp_path)
    responses: list[object] = []

    def submit(identifier: str) -> None:
        with TestClient(application) as client:
            contents = HEADER + ROW.replace("response-1", identifier)
            responses.append(client.post("/process", files=[("files", csv_file(contents=contents))]))

    threads = [threading.Thread(target=submit, args=(identifier,)) for identifier in ("one", "two")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert [response.status_code for response in responses] == [200, 200]  # type: ignore[attr-defined]
    assert all(artifact_names(response) == ARTIFACTS for response in responses)
    assert list((tmp_path / "requests").iterdir()) == []


def test_unexpected_processing_failure_is_safe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "umux_processor.api.run_pipeline_service",
        lambda *args, **kwargs: (_ for _ in ()).throw(PipelineServiceError("/private/path sensitive input")),
    )
    with TestClient(app(tmp_path)) as client:
        response = client.post("/process", files=[("files", csv_file())])

    assert response.status_code == 500
    assert response.json() == {"error": {"code": "PROCESSING_FAILED", "message": "The request could not be processed."}}
    assert "private" not in response.text
