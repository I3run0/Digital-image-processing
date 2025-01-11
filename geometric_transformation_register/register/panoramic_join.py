import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import argparse

def detect_and_compute_sift(gray):
    """
    Detect keypoints and compute descriptors using SIFT.
    
    Parameters:
    gray (numpy.ndarray): Grayscale image.
    
    Returns:
    tuple: keypoints and descriptors detected by SIFT.
    """
    sift = cv.SIFT_create(contrastThreshold=0.01)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def detect_and_compute_orb(gray):
    """
    Detect keypoints and compute descriptors using ORB.
    
    Parameters:
    gray (numpy.ndarray): Grayscale image.
    
    Returns:
    tuple: keypoints and descriptors detected by ORB.
    """
    detector = cv.ORB_create()
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints, descriptors

def detect_and_compute_brief(gray):
    """
    Detect keypoints using FAST and compute descriptors using BRIEF.
    
    Parameters:
    gray (numpy.ndarray): Grayscale image.
    
    Returns:
    tuple: keypoints and descriptors detected by BRIEF.
    """
    fast = cv.FastFeatureDetector_create()
    keypoints = fast.detect(gray, None)
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create(use_orientation=True)
    keypoints, descriptors = brief.compute(gray, keypoints)
    return keypoints, descriptors

KEYPOINT_DETECTOR = {
    'SIFT': detect_and_compute_sift,
    'ORB': detect_and_compute_orb,
    'BRIEF': detect_and_compute_brief
}

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
    argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Generate a panorama from two images")
    parser.add_argument("input_images_path", type=lambda x: x.split(','), help="Comma-separated input image paths")
    parser.add_argument("output_image_path", type=str, help="Output image path")
    parser.add_argument("-kpd", "--keypoint-detector", type=str, choices=KEYPOINT_DETECTOR.keys(), default='ORB', help="Keypoint detector")
    parser.add_argument("--show", action='store_true', help="Show images during processing")
    return parser.parse_args()

def load_and_convert_images(image1_path, image2_path):
    """
    Load images from given paths and convert them to grayscale.
    
    Parameters:
    image1_path (str): Path to the first image.
    image2_path (str): Path to the second image.
    
    Returns:
    tuple: Original images and their grayscale versions.
    """
    img1 = cv.imread(image1_path)
    img2 = cv.imread(image2_path)
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    return img1, img2, gray1, gray2

def match_descriptors(descriptors1, descriptors2):
    """
    Match descriptors between two images.
    
    Parameters:
    descriptors1 (numpy.ndarray): Descriptors of the first image.
    descriptors2 (numpy.ndarray): Descriptors of the second image.
    
    Returns:
    list: Sorted matches based on the distance.
    """
    bf = cv.BFMatcher(cv.NORM_HAMMING if descriptors1.dtype == np.uint8 else cv.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def estimate_homography(keypoints1, keypoints2, matches):
    """
    Estimate homography matrix using RANSAC.
    
    Parameters:
    keypoints1 (list): Keypoints from the first image.
    keypoints2 (list): Keypoints from the second image.
    matches (list): Matches between keypoints.
    
    Returns:
    numpy.ndarray: Homography matrix.
    """
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5)
    return H

def create_panorama(img1, img2, H):
    """
    Create a panorama by warping and combining two images.
    
    Parameters:
    img1 (numpy.ndarray): The first image.
    img2 (numpy.ndarray): The second image.
    H (numpy.ndarray): Homography matrix.
    
    Returns:
    numpy.ndarray: Panorama image.
    """
    height, width, _ = img2.shape
    panorama = cv.warpPerspective(img1, H, (width * 2, height))
    panorama[0:height, 0:width] = img2
    return panorama

def draw_matches(img1, keypoints1, img2, keypoints2, matches):
    """
    Draw lines between matched keypoints of two images.
    
    Parameters:
    img1 (numpy.ndarray): The first image.
    keypoints1 (list): Keypoints from the first image.
    img2 (numpy.ndarray): The second image.
    keypoints2 (list): Keypoints from the second image.
    matches (list): Matches between keypoints.
    
    Returns:
    numpy.ndarray: Image with drawn matches.
    """
    img_matches = cv.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img_matches

def show_image(image, title):
    """
    Display an image using matplotlib.
    
    Parameters:
    image (numpy.ndarray): Image to display.
    title (str): Title of the image.
    """
    plt.figure(figsize=(20, 10))
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(title)
    plt.show()

def save_image(image, path):
    """
    Save an image to the specified path.
    
    Parameters:
    image (numpy.ndarray): Image to save.
    path (str): Path where the image will be saved.
    """
    cv.imwrite(path, image)
    print(f"Image saved to {path}")

def main():
    """
    Main function to create a panorama from two images.
    
    Parameters:
    args (list): Command line arguments.
    """
    # Parse command line arguments
    args = parse_arguments()

    # Get input image paths
    image1_path, image2_path = args.input_images_path

    # (1) Load and convert images to grayscale
    img1, img2, gray1, gray2 = load_and_convert_images(image1_path, image2_path)

    # (2) Detect keypoints and compute descriptors
    keypoint_detector = KEYPOINT_DETECTOR[args.keypoint_detector]
    keypoints1, descriptors1 = keypoint_detector(gray1)
    keypoints2, descriptors2 = keypoint_detector(gray2)

    # (3) Match descriptors between the two images
    matches = match_descriptors(descriptors1, descriptors2)

    # (4) Select best matches
    good_matches = matches[:50]

    # (5) Estimate homography matrix using RANSAC
    H = estimate_homography(keypoints1, keypoints2, good_matches)

    # (6) Create panorama by warping and combining images
    panorama = create_panorama(img1, img2, H)
    
    # (7) Save and optionally show the panorama image
    save_image(panorama, args.output_image_path)
    if args.show:
        show_image(panorama, 'Panorama')

    # (8) Draw and save matches between keypoints
    img_matches = draw_matches(img1, keypoints1, img2, keypoints2, good_matches)
    matches_output_path = args.output_image_path.replace('.jpg', '_matches.jpg')  # Adjust extension as needed
    save_image(img_matches, matches_output_path)
    if args.show:
        show_image(img_matches, 'Correspondências de Pontos')

if __name__ == "__main__":
    main()
