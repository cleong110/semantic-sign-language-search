import pympi
from pathlib import Path
from pose_format import Pose

if __name__ == "__main__":
    # eaf = pympi.Elan.Eaf("/home/cleong/projects/sign_language/semantic-sign-language-search/videos_to_segments/out.eaf")
    eaf_path = Path('/home/cleong/projects/sign_language/semantic-sign-language-search/scripts/yt_asl_sample/4PtPO1g4LXA.eaf')
    eaf = pympi.Elan.Eaf(eaf_path)
    pose_path = eaf_path.with_suffix(".pose")
    with open(pose_path, "rb") as f:
        pose = Pose.read(f.read())

    print(eaf.tiers)


    
    for tier, values in eaf.tiers.items():
        aligned_annotations, reference_annotations, attributes, ordinal = values
        print (f"---------------------TIER {tier}--------------------")
        # {tier_name -> (aligned_annotations, reference_annotations, attributes, ordinal)},
        print(f"* ALIGNED ANNOTATIONS")
        print(aligned_annotations)

        for aligned_annotation_id, values in aligned_annotations.items():
            # aligned_annotations of the form: [{id -> (begin_ts, end_ts, value, svg_ref)}],
            begin_ts, end_ts, value, svg_ref = values
            print(begin_ts, end_ts)
            start_ms = eaf.timeslots[begin_ts]
            end_ms = eaf.timeslots[end_ts]
            print(start_ms, end_ms)

            

            # segmentation script has this conversion from frame to milliseconds, which we reverse
            # start_frame = int(segment["start"] / fps * 1000)
            # end_frame = int(segment["end"] / fps * 1000)
            # aka frames/second * 1000ms/second =
            milliseconds_per_frame = 1000*pose.body.fps
            start_frame = start_ms * pose.body.fps /1000
            end_frame = end_ms * pose.body.fps /1000
            frame_duration = end_frame-start_frame
            print(start_frame, end_frame, frame_duration)
        


        print("* REFERENCE ANNOTATIONS")
        print(reference_annotations)

        print("* ATTRIBUTES")
        print(attributes)
        print(attributes["TIER_ID"])

        print("* ORDINAL")
        print(ordinal)