import json
import zipfile

from scripts import download_coco_subset


def test_load_annotations_reads_official_archive_member(tmp_path, monkeypatch) -> None:
    archive_path = tmp_path / "annotations.zip"
    payload = {
        "images": [],
        "annotations": [],
        "categories": [],
    }
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            download_coco_subset.ANNOTATIONS_MEMBER,
            json.dumps(payload),
        )

    def fake_download(url, destination):
        assert url == download_coco_subset.ANNOTATIONS_URL
        destination.write_bytes(archive_path.read_bytes())

    monkeypatch.setattr(download_coco_subset, "_download_file", fake_download)

    assert download_coco_subset.load_annotations(
        None, tmp_path / "download.zip") == payload
    assert not (tmp_path / "download.zip").exists()


def test_download_subset_writes_ids_subset_and_selected_images(tmp_path, monkeypatch) -> None:
    coco = {
        "info": {},
        "licenses": [],
        "categories": [{"id": 1, "name": "item"}],
        "images": [
            {"id": 10, "file_name": "10.jpg", "width": 4, "height": 4},
            {"id": 20, "file_name": "20.jpg", "width": 4, "height": 4},
            {"id": 30, "file_name": "30.jpg", "width": 4, "height": 4},
        ],
        "annotations": [
            {"id": 101, "image_id": 10, "iscrowd": 0},
            {"id": 102, "image_id": 20, "iscrowd": 0},
            {"id": 103, "image_id": 30, "iscrowd": 1},
        ],
    }
    downloaded: list[str] = []

    def fake_download_image(image, output_dir):
        path = output_dir / image["file_name"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\xff\xd8\xff")
        downloaded.append(image["file_name"])
        return path

    monkeypatch.setattr(download_coco_subset,
                        "download_image", fake_download_image)
    ids_path = tmp_path / "selected_image_ids.json"

    image_count, annotation_count = download_coco_subset.download_subset(
        coco=coco,
        output_dir=tmp_path / "coco_subset",
        ids_output=ids_path,
        count=2,
        seed=42,
        min_instances=1,
        max_instances=1,
        workers=2,
    )

    selected_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    assert image_count == 2
    assert annotation_count == 2
    assert len(downloaded) == 2
    subset = json.loads(
        (tmp_path / "coco_subset/annotations/subset.json").read_text(encoding="utf-8")
    )
    assert {image["id"] for image in subset["images"]} == set(selected_ids)
    assert {annotation["image_id"]
            for annotation in subset["annotations"]} == set(selected_ids)
