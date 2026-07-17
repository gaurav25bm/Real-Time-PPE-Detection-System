import cv2
from detector import detect_frame


def process_video(input_path, output_path):

    print("===================================")
    print("Opening video...")

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise Exception("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output resolution
    output_width = 1280
    output_height = 720

    print(f"Output Size : {output_width} x {output_height}")
    print(f"FPS         : {fps}")
    print(f"Frames      : {total_frames}")

    fourcc = cv2.VideoWriter_fourcc(*"avc1")

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (output_width, output_height)
    )

    if not out.isOpened():
        raise Exception("VideoWriter could not be opened.")

    frame_no = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_no += 1

        if frame_no % 10 == 0:
            print(f"Processing {frame_no}/{total_frames}")

        # Resize before detection (faster)
        frame = cv2.resize(frame, (640, 640))

        annotated, _ = detect_frame(frame)

        # Resize back to output size
        annotated = cv2.resize(
            annotated,
            (output_width, output_height)
        )

        annotated = annotated.astype("uint8")

        out.write(annotated)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video saved successfully.")
    print("===================================")

    return output_path