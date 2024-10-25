import os
from dotenv import load_dotenv
import cv2
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from swarms import Agent
from time import sleep
from typing import Optional
from loguru import logger
from tenacity import retry, wait_fixed, stop_after_attempt
from model import VisionAPIWrapper

# Load environment variables
load_dotenv()

# Set up logging
logger.add("dj_agent_log.log", rotation="1 MB", level="DEBUG")

# Initialize the Spotify API for music control
spotify: spotipy.Spotify = spotipy.Spotify(
    auth_manager=SpotifyOAuth(
        client_id=os.environ.get("SPOTIFY_CLIENT_ID"),
        client_secret=os.environ.get("SPOTIFY_CLIENT_SECRET"),
        redirect_uri="http://localhost:8080",
        scope="user-read-playback-state user-modify-playback-state",
    )
)

# Initialize the vision model for crowd analysis (GPT-4 Vision or OpenCV)
vision_llm = VisionAPIWrapper(
    api_key="",
    max_tokens=500,
)

# Define the crowd analysis task for acid techno selection
task: str = (
    "Analyze this real-time image of a crowd and determine the overall energy level "
    "based on body movement, facial expressions, and crowd density. "
    "If the crowd appears highly energetic, output a high-intensity acid techno song "
    "with a fast tempo and heavy bass. If the energy is lower, output a slightly slower "
    "but still engaging acid techno song. Provide the song recommendation to match the "
    "crowd’s energy, focusing on high-energy acid techno music."
)


# Retry decorator to handle API call failures
@retry(wait=wait_fixed(5), stop=stop_after_attempt(3))
def get_spotify_track(track_name: str) -> Optional[str]:
    """
    Fetches a track from Spotify by its name.
    Retries up to 3 times with a 5-second wait in case of failure.

    :param track_name: The name of the track to search for.
    :return: Spotify track ID if successful, None otherwise.
    """
    logger.info(f"Fetching Spotify track for: {track_name}")
    try:
        results = spotify.search(q=track_name, type="track", limit=1)
        track = results["tracks"]["items"][0]["id"]
        logger.debug(
            f"Found track ID: {track} for track name: {track_name}"
        )
        return track
    except Exception as e:
        logger.error(f"Error fetching Spotify track: {e}")
        raise


def analyze_crowd(frame: str) -> str:
    """
    Analyze the provided image frame of the crowd and return a description
    of the energy level and a song recommendation.

    :param frame: The path to the image frame to analyze.
    :return: A string containing the recommended song based on crowd energy.
    """
    logger.info("Analyzing crowd energy from image.")
    try:
        # Initialize the workflow agent
        agent: Agent = Agent(
            agent_name="AcidTechnoDJ_CrowdAnalyzer",
            # system_prompt=task,
            llm=vision_llm,
            max_loops=1,
            # autosave=True,
            # dashboard=True,
            # multi_modal=True,
        )

        # Analyze the frame
        response: str = agent.run(task, frame)
        logger.debug(f"Crowd analysis result: {response}")
        return response  # This should be a recommended song title
    except Exception as e:
        logger.error(f"Error analyzing crowd: {e}")
        raise


def play_song(song_name: str) -> None:
    """
    Fetches the song by name from Spotify and plays it.

    :param song_name: The name of the song to play.
    """
    logger.info(f"Attempting to play song: {song_name}")
    track_id: Optional[str] = get_spotify_track(song_name)
    if track_id:
        spotify.start_playback(uris=[f"spotify:track:{track_id}"])
        logger.info(f"Now playing track: {track_id}")
    else:
        logger.error(f"Could not play song: {song_name}")


def save_frame(frame: cv2.Mat, path: str = "temp_frame.jpg") -> str:
    """
    Saves a frame from the video feed as an image file.

    :param frame: The OpenCV frame (image) to save.
    :param path: The file path to save the image to (default: 'temp_frame.jpg').
    :return: The path to the saved image file.
    """
    cv2.imwrite(path, frame)
    logger.info(f"Frame saved to {path}")
    return path


@retry(wait=wait_fixed(5), stop=stop_after_attempt(3))
def capture_video_feed() -> Optional[cv2.VideoCapture]:
    """
    Initializes and returns the video capture feed.

    :return: OpenCV video capture object if successful, None otherwise.
    """
    logger.info("Attempting to capture video feed.")
    try:
        camera: cv2.VideoCapture = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise RuntimeError("Failed to open camera.")
        logger.debug("Camera feed opened successfully.")
        return camera
    except Exception as e:
        logger.error(f"Error capturing video feed: {e}")
        raise


def run_dj_agent():
    """
    Runs the DJ agent in a loop, analyzing the crowd's energy level in real-time,
    recommending acid techno tracks based on the analysis, and playing the recommended
    song if applicable.
    """
    logger.info("Starting Acid Techno DJ Agent...")

    # Capture the video feed
    camera: Optional[cv2.VideoCapture] = capture_video_feed()

    try:
        while True:
            # Capture a frame every 5 seconds
            ret, frame = camera.read()
            if not ret:
                logger.warning(
                    "Failed to capture frame from video feed."
                )
                break

            # Save the frame as an image file
            frame_path: str = save_frame(frame)

            # Analyze the current crowd state and get a song recommendation
            recommended_song: str = analyze_crowd(frame_path)
            logger.info(f"Recommended song: {recommended_song}")

            # Play the recommended song based on the analysis
            play_song(recommended_song)

            # Wait for 5 seconds before capturing the next frame
            sleep(5)

    except Exception as e:
        logger.error(f"DJ Agent encountered an error: {e}")
    finally:
        # Release the camera feed when done
        if camera:
            camera.release()
        cv2.destroyAllWindows()
        logger.info("DJ Agent has stopped.")


# Run the DJ agent
if __name__ == "__main__":
    run_dj_agent()
