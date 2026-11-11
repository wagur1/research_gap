"""Download Kaggle outputs into an empty directory with explicit UTF-8 logs."""
from pathlib import Path


def download_kernel_outputs(api, kernel_id, destination):
    import requests
    from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest

    destination = Path(destination).resolve()
    if not destination.is_dir() or any(destination.iterdir()):
        raise ValueError("Kernel download requires a new empty directory")
    owner, slug, _ = api.parse_kernel_string(kernel_id)
    token = None
    seen_tokens = set()
    while True:
        with api.build_kaggle_client() as client:
            request = ApiListKernelSessionOutputRequest()
            request.user_name = owner
            request.kernel_slug = slug
            api._set_paging(request, 20, token)
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
        for item in response.files or []:
            relative = Path(item.file_name)
            target = destination / relative
            if relative.is_absolute() or ".." in relative.parts or ":" in str(relative) or not target.resolve().is_relative_to(destination):
                raise ValueError("Unsafe kernel output filename")
            target.parent.mkdir(parents=True, exist_ok=True)
            with requests.get(item.url, stream=True, timeout=(30, 180)) as result:
                result.raise_for_status()
                with target.open("xb") as handle:
                    for chunk in result.iter_content(1024 * 1024):
                        handle.write(chunk)
        if response.log:
            with (destination / f"{slug}.log").open("a", encoding="utf-8", newline="") as handle:
                handle.write(response.log)
        token = response.next_page_token
        if not token:
            return
        if token in seen_tokens:
            raise ValueError("Repeated output pagination token")
        seen_tokens.add(token)
