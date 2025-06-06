import time
import requests
import os

# replace your vercel domain
base_url = "https://suno-api-eight-psi.vercel.app/"


def custom_generate_audio(payload):
    url = f"{base_url}/api/custom_generate"
    response = requests.post(
        url,
        json=payload,
        headers={"Content-Type": "application/json"},
    )
    return response.json()


def extend_audio(payload):
    url = f"{base_url}/api/extend_audio"
    response = requests.post(
        url,
        json=payload,
        headers={"Content-Type": "application/json"},
    )
    return response.json()


def generate_audio_by_prompt(payload):
    url = f"{base_url}/api/generate"
    response = requests.post(
        url,
        json=payload,
        headers={"Content-Type": "application/json"},
    )
    return response.json()


def get_audio_information(audio_ids):
    url = f"{base_url}/api/get?ids={audio_ids}"
    response = requests.get(url)
    return response.json()


# def get_quota_information():
#     url = f"{base_url}/api/get_limit"
#     response = requests.get(url)
#     return response.json()


def get_clip(clip_id):
    url = f"{base_url}/api/clip?id={clip_id}"
    response = requests.get(url)
    return response.json()


if __name__ == "__main__":
    # Create a directory for downloaded audio files
    download_dir = "downloaded_audio"
    os.makedirs(
        download_dir, exist_ok=True
    )  # Create directory if it doesn't exist

    data = generate_audio_by_prompt(
        {
            "prompt": (
                "A popular heavy metal song about war, sung by a deep-voiced male singer, slowly and melodiously. The lyrics depict the sorrow of people after the war."
            ),
            "make_instrumental": False,
            "wait_audio": False,
        }
    )

    ids = f"{data[0]['id']},{data[1]['id']}"
    print(f"ids: {ids}")

    for _ in range(60):
        data = get_audio_information(ids)
        if data[0]["status"] == "streaming":
            for clip in data:
                print(f"{clip['id']} ==> {clip['audio_url']}")

                # Download the clip as an MP3
                response = requests.get(clip["audio_url"])
                if response.status_code == 200:
                    filename = os.path.join(
                        download_dir, f"{clip['id']}.mp3"
                    )  # Save in the new directory
                    with open(filename, "wb") as f:
                        f.write(response.content)
                    print(f"Downloaded: {filename}")
                else:
                    print(f"Failed to download clip {clip['id']}")

            break
        # sleep 5s
        time.sleep(5)
