"""modal_speedtest.py -- measure Modal's upload/download throughput, to compare
against the DigitalOcean droplet and this Mac before choosing where to push from."""
import modal

app = modal.App("speedtest")
image = modal.Image.debian_slim(python_version="3.11").apt_install("curl")


@app.function(image=image, timeout=900)
def test():
    import subprocess
    subprocess.run("dd if=/dev/urandom of=/tmp/up.bin bs=1M count=200 2>/dev/null",
                   shell=True, check=True)
    up = subprocess.run(
        "curl -s -o /dev/null -w '%{speed_upload} %{time_total}' "
        "-T /tmp/up.bin https://speed.cloudflare.com/__up",
        shell=True, capture_output=True, text=True).stdout
    down = subprocess.run(
        "curl -s -o /dev/null -w '%{speed_download}' "
        "'https://speed.cloudflare.com/__down?bytes=200000000'",
        shell=True, capture_output=True, text=True).stdout
    # also time a real GitHub-bound handshake+fetch, since GitHub ingest -- not
    # raw bandwidth -- is what actually limited the Mac (1.9MB/s raw, 0.33 to GH)
    gh = subprocess.run(
        "curl -s -o /dev/null -w '%{speed_download}' "
        "https://codeload.github.com/git/git/tar.gz/refs/tags/v2.43.0",
        shell=True, capture_output=True, text=True).stdout
    return {"upload_Bps": up, "download_Bps": down, "github_down_Bps": gh}


@app.local_entrypoint()
def main():
    r = test.remote()
    u = r["upload_Bps"].split()
    print(f"MODAL upload:   {float(u[0])/1e6:.2f} MB/s ({u[1]}s for 200MB)")
    print(f"MODAL download: {float(r['download_Bps'])/1e6:.2f} MB/s")
    print(f"MODAL <- github.com: {float(r['github_down_Bps'])/1e6:.2f} MB/s")
