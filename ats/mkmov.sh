fps=10
ffmpeg -y -r $fps -pattern_type glob -i "../recflows/pngs/*.png" -vcodec libx264 -crf 18 -pix_fmt yuv420p out.mp4
