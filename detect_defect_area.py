import cv2
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

images_dir = os.path.join(BASE_DIR, "images")
result_dir = os.path.join(BASE_DIR, "results")
os.makedirs(result_dir, exist_ok=True)

valid_ext = (".jpg", ".jpeg", ".png", ".bmp")

for filename in os.listdir(images_dir):

    if not filename.lower().endswith(valid_ext):
        continue

    img_path = os.path.join(images_dir, filename)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Could not load: {filename}")
        continue

    img = cv2.resize(img, (600, 400))
    output = img.copy()

    margin = 20
    roi = img[margin:-margin, margin:-margin]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Detect pale/white damaged areas
    lower_white = np.array([0, 0, 135])
    upper_white = np.array([180, 130, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Remove long horizontal and vertical panel grid lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 45))

    horizontal_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vertical_kernel)

    grid_lines = cv2.bitwise_or(horizontal_lines, vertical_lines)
    mask_without_grid = cv2.subtract(mask, grid_lines)

    # Clean and connect damaged regions
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_medium = np.ones((11, 11), np.uint8)

    mask_without_grid = cv2.morphologyEx(
        mask_without_grid,
        cv2.MORPH_OPEN,
        kernel_small
    )

    mask_without_grid = cv2.morphologyEx(
        mask_without_grid,
        cv2.MORPH_CLOSE,
        kernel_medium
    )

    mask_without_grid = cv2.dilate(mask_without_grid, kernel_small, iterations=1)

    contours, _ = cv2.findContours(
        mask_without_grid,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    defect_count = 0
    total_defect_area = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 250:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        aspect_ratio = w / h

        # Ignore very long thin shapes
        if aspect_ratio > 6 or aspect_ratio < 0.16:
            continue

        rect_area = w * h
        fill_ratio = area / rect_area

        # Ignore weak/noisy detections
        if fill_ratio < 0.08:
            continue

        defect_count += 1
        total_defect_area += area

        x += margin
        y += margin

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2
        )

    if defect_count == 0:
        status = "OK"
        severity = "NONE"
    elif total_defect_area < 1500:
        status = "DEFECTIVE"
        severity = "LOW"
    elif total_defect_area < 6000:
        status = "DEFECTIVE"
        severity = "MEDIUM"
    else:
        status = "DEFECTIVE"
        severity = "HIGH"

    cv2.putText(output, f"Status: {status}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.putText(output, f"Severity: {severity}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.putText(output, f"Detected Areas: {defect_count}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    result_path = os.path.join(result_dir, f"result_{filename}")
    cv2.imwrite(result_path, output)

    # Save mask too, very useful for debugging
    mask_path = os.path.join(result_dir, f"mask_{filename}")
    cv2.imwrite(mask_path, mask_without_grid)

    print(
        f"Processed: {filename} | "
        f"Status: {status} | "
        f"Severity: {severity} | "
        f"Defects: {defect_count} | "
        f"Area: {total_defect_area}"
    )

    cv2.imshow("Defect Detection", output)
    cv2.imshow("Mask", mask_without_grid)
    cv2.waitKey(0)

cv2.destroyAllWindows()