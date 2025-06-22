import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time 
from math import cos, sin, radians, degrees, asin, sqrt
import warnings
warnings.filterwarnings('ignore')

class StarMatcher:
    def __init__(self, catalog_path):
        self.catalog = self.load_catalog(catalog_path)
        self.detected_stars = []
        self.original_image = None

    def hms_to_degrees(self, h, m, s):
        return h * 15.0 + m * 0.25 + s * (0.25/60.0)

    def dms_to_degrees(self, sign, d, m, s):
        sign_val = -1 if str(sign).strip() == '-' else 1
        return sign_val * (d + m/60.0 + s/3600.0)

    def load_catalog(self, path):
        cat = pd.read_csv(path)
        if all(col in cat.columns for col in ['RAh', 'RAm', 'RAs', 'DE-', 'DEd', 'DEm', 'DEs']):
            cat['RA'] = cat.apply(lambda r: self.hms_to_degrees(r['RAh'], r['RAm'], r['RAs']), axis=1)
            cat['DEC'] = cat.apply(lambda r: self.dms_to_degrees(r['DE-'], r['DEd'], r['DEm'], r['DEs']), axis=1)
            cat['magnitude'] = pd.to_numeric(cat.get('Vmag', 5.0), errors='coerce').fillna(5.0)
            cat['name'] = cat.get('Name', '').fillna('').astype(str).combine_first(cat.get('HR', '').astype(str))
        else:
            cat = cat.rename(columns={
                'RAJ2000': 'RA',
                'DEJ2000': 'DEC',
                'Vmag': 'magnitude',
                'Name': 'name'
            })
        return cat.dropna(subset=['RA', 'DEC'])

    def detect_stars(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Failed to load image")
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 50
        params.maxThreshold = 255
        params.filterByArea = True
        params.minArea = 3
        params.maxArea = 3000
        params.filterByCircularity = True
        params.minCircularity = 0.1
        params.filterByColor = True
        params.blobColor = 255
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(blur)
        self.original_image = image
        self.detected_stars = []
        for i, kp in enumerate(keypoints):
            x, y = int(kp.pt[0]), int(kp.pt[1])
            r = int(max(5, kp.size / 2))
            star_region = image[max(y-r,0):min(y+r,image.shape[0]-1), max(x-r,0):min(x+r,image.shape[1]-1)]
            brightness = np.mean(star_region)
            if brightness < 80:
                continue
            self.detected_stars.append({
                'id': i+1, 'x': x, 'y': y, 'radius': int(kp.size/2),
                'brightness': brightness,
                'magnitude': 8.0 - 2.5 * np.log10(brightness / 100.0)
            })
        return self.detected_stars

    def pixel_to_celestial(self, x, y, shape, fov=20, ra0=0, dec0=0):
        h, w = shape
        rx = (x - w/2) / w
        ry = (h/2 - y) / h
        dra = rx * fov / cos(radians(dec0))
        ddec = ry * fov
        return ra0 + dra, dec0 + ddec

    def angular_distance(self, ra1, dec1, ra2, dec2):
        ra1, dec1, ra2, dec2 = map(radians, [ra1, dec1, ra2, dec2])
        a = sin((dec2-dec1)/2)**2 + cos(dec1)*cos(dec2)*sin((ra2-ra1)/2)**2
        return degrees(2 * asin(sqrt(a)))

    def match_stars_by_pattern(self, fov=20, ra0=0, dec0=0, max_mag=7.5):
        matches = []
        detected_celestial = [{
            'id': s['id'],
            'pixel_x': s['x'],
            'pixel_y': s['y'],
            'magnitude': s['magnitude'],
            'ra': self.pixel_to_celestial(s['x'], s['y'], self.original_image.shape, fov, ra0, dec0)[0],
            'dec': self.pixel_to_celestial(s['x'], s['y'], self.original_image.shape, fov, ra0, dec0)[1]
        } for s in self.detected_stars]

        catalog = self.catalog[
            (self.catalog['magnitude'] <= max_mag) &
            (self.catalog['RA'] >= ra0 - fov/2) & (self.catalog['RA'] <= ra0 + fov/2) &
            (self.catalog['DEC'] >= dec0 - fov/2) & (self.catalog['DEC'] <= dec0 + fov/2)
        ]

        for d in detected_celestial:
            best = None
            min_score = float('inf')
            for _, c in catalog.iterrows():
                dist = self.angular_distance(d['ra'], d['dec'], c['RA'], c['DEC'])
                score = dist
                if score < min_score:
                    best = {
                        'catalog_star': c,
                        'distance': dist
                    }
                    min_score = score
            if best:
                matches.append({
                    'detected_star': d,
                    'catalog_match': best,
                    'confidence': 1.0 / (1.0 + min_score)
                })
        return matches

def visualize_matches_with_confidence(matcher, matches, total_time, save_path=None):
    if not matches:
        print("No matches to visualize")
        return
    result = cv2.cvtColor(matcher.original_image, cv2.COLOR_GRAY2BGR)

    green_matches = 0
    yellow_matches = 0
    red_matches = 0

    for star in matcher.detected_stars:
        cv2.circle(result, (star['x'], star['y']), star['radius'], (255, 255, 0), 1)
        cv2.putText(result, str(star['id']), (star['x'] + 8, star['y'] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

    match_details_for_file = []

    for match in matches:
        det = match['detected_star']
        conf = match['confidence']
        catalog_star = match['catalog_match']['catalog_star']

        color = (0, 0, 255) # Default Red for < 0.5
        color_name = "Red"
        if conf >= 0.75:
            color = (0, 255, 0) # Green
            color_name = "Green"
            green_matches += 1
        elif conf >= 0.5:
            color = (0, 255, 255) # Yellow
            color_name = "Yellow"
            yellow_matches += 1
        else:
            red_matches += 1

        cv2.circle(result, (det['pixel_x'], det['pixel_y']), 10, color, 2)
        cv2.putText(result, f"{conf:.2f}", (det['pixel_x'] + 5, det['pixel_y'] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        match_details_for_file.append(
            f"Detected Star ID: {det['id']}, Pixel Coords: ({det['pixel_x']}, {det['pixel_y']}), "
            f"Detected Magnitude: {det['magnitude']:.2f}\n"
            f"  Catalog Star Name: {catalog_star.get('name', 'N/A')}, "
            f"Catalog RA: {catalog_star['RA']:.2f}, Catalog DEC: {catalog_star['DEC']:.2f}, "
            f"Catalog Magnitude: {catalog_star['magnitude']:.2f}\n"
            f"  Confidence: {conf:.2f} ({color_name} match)\n"
            f"  Angular Distance: {match['catalog_match']['distance']:.4f} degrees\n"
            f"----------------------------------------------------\n"
        )

    # Add match counts and time to the title
    time_status = "SUCCESS" if total_time < 1.0 else "FAILURE"
    title_text = (
        f"Matched Stars with Confidence Colors\n"
        f"Detected: {len(matcher.detected_stars)}, Total Matches: {len(matches)}\n"
        f"Green: {green_matches}, Yellow: {yellow_matches}, Red: {red_matches}\n"
        f"Matching Time: {total_time:.4f} seconds ({time_status}: < 1 sec)"
    )

    plt.figure(figsize=(12, 10))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title(title_text)
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    txt_save_path = save_path.replace('.png', '.txt') if save_path else "matched_output.txt"
    with open(txt_save_path, 'w') as f:
        f.write(f"--- Star Matching Results ---\n")
        f.write(f"Total Detected Stars: {len(matcher.detected_stars)}\n")
        f.write(f"Total Matches Found: {len(matches)}\n")
        f.write(f"Matches by Confidence Level:\n")
        f.write(f"  Green (Confidence >= 0.75): {green_matches}\n")
        f.write(f"  Yellow (0.5 <= Confidence < 0.75): {yellow_matches}\n")
        f.write(f"  Red (Confidence < 0.5): {red_matches}\n")
        f.write(f"\n--- Performance ---\n")
        f.write(f"Matching Time: {total_time:.4f} seconds\n")
        f.write(f"Time Requirement (< 1 sec): {time_status}\n")
        f.write(f"\n--- Detailed Match Information ---\n\n")
        for detail in match_details_for_file:
            f.write(detail)
    print(f"Match details saved to {txt_save_path}")

    plt.show()



def main():
    catalog_path = '/home/ben/Desktop/Space_Engineering/yale_bright_star_catalog_v5.csv'
    image_path = '/home/ben/Desktop/Space_Engineering/Assignment1/Photos_of_Stars/IMG_3046.jpg'
    output_dir = '/home/ben/Desktop/Space_Engineering/Assignment1/'

    matcher = StarMatcher(catalog_path)

    # Grid search
    max_magnitude = 7.5
    best_result = None
    print("\n🔄 Running grid search for best RA/DEC/FOV...")

    matcher.detect_stars(image_path) 

    # First run, finding the best value for fov,ra,dec
    for fov in range(19, 23, 1):  
        for ra in range(30, 60, 5):  
            for dec in range(15, 45, 5): 
                matches = matcher.match_stars_by_pattern(
                    fov=fov,
                    ra0=ra,
                    dec0=dec,
                    max_mag=max_magnitude
                )

                bad = sum(1 for m in matches if m['confidence'] < 0.5)
                avg_conf = np.mean([m['confidence'] for m in matches]) if matches else 0

                print(f"FOV={fov}, RA={ra}°, DEC={dec}° → {len(matches)} matches ({bad} bad), avg confidence: {avg_conf:.3f}")

                if best_result is None or bad < best_result['bad_matches']:
                    best_result = {
                        'fov': fov,
                        'ra': ra,
                        'dec': dec,
                        'bad_matches': bad,
                        'matches': matches,
                        'avg_conf': avg_conf
                    }

    print("\nBest Parameters Found:")
    print(f"FOV: {best_result['fov']}°, RA: {best_result['ra']}°, DEC: {best_result['dec']}°")
    print(f"Matches: {len(best_result['matches'])}, Bad Matches: {best_result['bad_matches']}, Avg Confidence: {best_result['avg_conf']:.3f}")

    # Actual run
    print("\n⏱ Measuring final execution time on best parameters...")
    start_time = time.time()

    matcher.detect_stars(image_path) 
    best_matches = matcher.match_stars_by_pattern(
        fov=best_result['fov'],
        ra0=best_result['ra'],
        dec0=best_result['dec'],
        max_mag=max_magnitude
    )

    end_time = time.time()
    total_time = end_time - start_time

    print(f"Final Matching Time: {total_time:.4f} seconds")
    if total_time < 1.0:
        print(" Time requirement met (< 1 second)")
    else:
        print(" Time requirement not met (> 1 second)")

    # Visualization
    visualize_matches_with_confidence(matcher, best_matches, total_time, save_path=output_dir + 'best_star_matches.png')
    matcher.save_matches(best_matches, output_dir + 'best_star_matches.csv')




if __name__ == "__main__":
    main()

