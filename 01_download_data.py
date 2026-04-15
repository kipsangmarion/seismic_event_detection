"""
01_download_data.py

Downloads raw seismic waveforms from IRIS for two classes:
  - Earthquake: 5-min windows around known USGS events
  - Noise:      5-min windows at random times with no known events

Outputs:
  data/earthquake/event_<i>.mseed
  data/noise/noise_<i>.mseed
  earthquake_catalog.xml
"""

import os
import random
import time
from dotenv import load_dotenv
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

load_dotenv()

# Config
IRIS_CLIENT        = os.getenv("IRIS_CLIENT", "IRIS")
NETWORK            = os.getenv("NETWORK", "IU")
STATION            = os.getenv("STATION", "ANMO")
LOCATION           = os.getenv("LOCATION", "00")
CHANNEL            = os.getenv("CHANNEL", "HHZ")

START_DATE         = os.getenv("START_DATE", "2023-01-01")
END_DATE           = os.getenv("END_DATE", "2024-12-31")
MIN_MAGNITUDE      = float(os.getenv("MIN_MAGNITUDE", 4.0))
MAX_MAGNITUDE      = float(os.getenv("MAX_MAGNITUDE", 7.0))
EVENT_LIMIT        = int(os.getenv("EVENT_LIMIT", 300))
WAVEFORM_PRE_SEC   = int(os.getenv("WAVEFORM_PRE_SECONDS", 60))
WAVEFORM_POST_SEC  = int(os.getenv("WAVEFORM_POST_SECONDS", 240))
NOISE_COUNT        = int(os.getenv("NOISE_COUNT", 200))
NOISE_WINDOW_SEC   = WAVEFORM_PRE_SEC + WAVEFORM_POST_SEC  # 300 s, same length as earthquake windows

EARTHQUAKE_DIR     = os.getenv("EARTHQUAKE_DIR", "data/earthquake")
NOISE_DIR          = os.getenv("NOISE_DIR", "data/noise")
CATALOG_FILE       = os.getenv("CATALOG_FILE", "earthquake_catalog.xml")

TARGET_EARTHQUAKE  = 200   # stop early once we have this many
TARGET_NOISE       = 200


# Helpers
def make_dirs():
    os.makedirs(EARTHQUAKE_DIR, exist_ok=True)
    os.makedirs(NOISE_DIR, exist_ok=True)


def count_existing(directory, prefix, ext=".mseed"):
    """Return sorted list of existing file indices in a directory."""
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(ext)]
    return len(files)


def random_noise_times(n, start, end, seed=42):
    """Generate n random UTCDateTimes uniformly distributed in [start, end]."""
    random.seed(seed)
    span = end - start
    return [start + random.uniform(0, span) for _ in range(n)]


# Phase 1a: Download earthquake catalog
def download_catalog(client):
    if os.path.exists(CATALOG_FILE):
        print(f"[catalog] Found existing {CATALOG_FILE} — skipping download.")
        from obspy import read_events
        return read_events(CATALOG_FILE)

    print(f"[catalog] Querying USGS for M{MIN_MAGNITUDE}–{MAX_MAGNITUDE} events "
          f"({START_DATE} to {END_DATE}, limit={EVENT_LIMIT}) ...")
    catalog = client.get_events(
        starttime=UTCDateTime(START_DATE),
        endtime=UTCDateTime(END_DATE),
        minmagnitude=MIN_MAGNITUDE,
        maxmagnitude=MAX_MAGNITUDE,
        limit=EVENT_LIMIT,
    )
    catalog.write(CATALOG_FILE, format="QUAKEML")
    print(f"[catalog] {len(catalog)} events saved to {CATALOG_FILE}")
    return catalog


# Phase 1b: Download earthquake waveforms
def download_earthquakes(client, catalog):
    existing = count_existing(EARTHQUAKE_DIR, "event_")
    if existing >= TARGET_EARTHQUAKE:
        print(f"[earthquakes] Already have {existing} files — skipping.")
        return

    print(f"[earthquakes] Starting downloads (have {existing}, target {TARGET_EARTHQUAKE}) ...")
    downloaded = existing
    skipped    = 0

    for i, event in enumerate(catalog):
        if downloaded >= TARGET_EARTHQUAKE:
            break

        out_path = os.path.join(EARTHQUAKE_DIR, f"event_{i}.mseed")
        if os.path.exists(out_path):
            downloaded += 1
            continue

        origin_time = event.origins[0].time
        try:
            st = client.get_waveforms(
                network=NETWORK,
                station=STATION,
                location=LOCATION,
                channel=CHANNEL,
                starttime=origin_time - WAVEFORM_PRE_SEC,
                endtime=origin_time + WAVEFORM_POST_SEC,
            )
            st.write(out_path, format="MSEED")
            downloaded += 1
            print(f"  [{downloaded:>3}/{TARGET_EARTHQUAKE}] event_{i}.mseed  "
                  f"(M{event.magnitudes[0].mag:.1f}, {origin_time.date})")
        except Exception as e:
            skipped += 1
            print(f"  [skip] event_{i}: {e}")

        # polite delay to avoid hammering IRIS
        time.sleep(0.5)

    print(f"[earthquakes] Done — {downloaded} downloaded, {skipped} skipped.")


# Phase 1c: Download noise waveforms
def download_noise(client):
    existing = count_existing(NOISE_DIR, "noise_")
    if existing >= TARGET_NOISE:
        print(f"[noise] Already have {existing} files — skipping.")
        return

    print(f"[noise] Starting downloads (have {existing}, target {TARGET_NOISE}) ...")

    # Sample from the full date range to match earthquake windows
    noise_times = random_noise_times(
        n=TARGET_NOISE * 2,            # oversample to account for failures
        start=UTCDateTime(START_DATE),
        end=UTCDateTime(END_DATE),
    )

    downloaded = existing
    skipped    = 0
    idx        = 0

    for t in noise_times:
        if downloaded >= TARGET_NOISE:
            break

        out_path = os.path.join(NOISE_DIR, f"noise_{idx}.mseed")
        if os.path.exists(out_path):
            downloaded += 1
            idx += 1
            continue

        try:
            st = client.get_waveforms(
                network=NETWORK,
                station=STATION,
                location=LOCATION,
                channel=CHANNEL,
                starttime=t,
                endtime=t + NOISE_WINDOW_SEC,
            )
            st.write(out_path, format="MSEED")
            downloaded += 1
            idx += 1
            print(f"  [{downloaded:>3}/{TARGET_NOISE}] noise_{idx-1}.mseed  ({t.date})")
        except Exception as e:
            skipped += 1
            print(f"  [skip] noise at {t.date}: {e}")

        time.sleep(0.5)

    print(f"[noise] Done — {downloaded} downloaded, {skipped} skipped.")


# Main
def main():
    make_dirs()

    print("=" * 60)
    print("Connecting to IRIS ...")
    client = Client(IRIS_CLIENT)
    print("Connected.\n")

    catalog = download_catalog(client)
    print()
    download_earthquakes(client, catalog)
    print()
    download_noise(client)

    print("\n" + "=" * 60)
    eq_count    = count_existing(EARTHQUAKE_DIR, "event_")
    noise_count = count_existing(NOISE_DIR, "noise_")
    print(f"Final counts:")
    print(f"  Earthquake waveforms : {eq_count}")
    print(f"  Noise waveforms      : {noise_count}")
    print(f"  Total                : {eq_count + noise_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
