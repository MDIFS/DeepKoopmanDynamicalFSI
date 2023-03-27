fps=30
ffmpeg -y -r $fps -pattern_type glob -i "u4.0/u4/*.png" -vcodec libx264 -crf 18 -pix_fmt yuv420p out.mp4
