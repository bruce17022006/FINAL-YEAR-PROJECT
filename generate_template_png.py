import os
from PIL import Image, ImageDraw, ImageFont

def create_template_image(output_path="architecture_template.png"):
    width, height = 1400, 580
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Fonts
    try:
        font_title = ImageFont.truetype("arial.ttf", 15)
        font_body = ImageFont.truetype("arial.ttf", 11)
        font_bold = ImageFont.truetype("arialbd.ttf", 11)
        font_small = ImageFont.truetype("arial.ttf", 10)
    except:
        font_title = font_body = font_bold = font_small = ImageFont.load_default()

    # Outer rounded rectangle
    draw.rounded_rectangle([15, 15, width - 15, height - 15], radius=35, outline="black", width=2, fill="white")

    # Title
    title = "Dynamic Knowledge Distillation Transfer"
    draw.text((width // 2, 45), title, fill="black", font=font_title, anchor="mm")

    def draw_box(x, y, w, h, text_lines, rounded=False, radius=8, left_align=False):
        if rounded:
            draw.rounded_rectangle([x, y, x + w, y + h], radius=radius, outline="black", width=1, fill="white")
        else:
            draw.rectangle([x, y, x + w, y + h], outline="black", width=1, fill="white")
        
        line_height = 14
        total_text_h = len(text_lines) * line_height
        start_y = y + (h - total_text_h) // 2
        for i, line in enumerate(text_lines):
            ly = start_y + i * line_height
            if left_align:
                draw.text((x + 10, ly), line, fill="black", font=font_body)
            else:
                draw.text((x + w // 2, ly), line, fill="black", font=font_body, anchor="mt")

    def draw_arrow(p1, p2, label=None, label_pos=None):
        draw.line([p1, p2], fill="black", width=1)
        # Arrowhead
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        import math
        angle = math.atan2(dy, dx)
        arr_len = 8
        a1 = angle + math.pi - 0.4
        a2 = angle + math.pi + 0.4
        tip_x, tip_y = p2[0], p2[1]
        x1 = tip_x + arr_len * math.cos(a1)
        y1 = tip_y + arr_len * math.sin(a1)
        x2 = tip_x + arr_len * math.cos(a2)
        y2 = tip_y + arr_len * math.sin(a2)
        draw.polygon([(tip_x, tip_y), (x1, y1), (x2, y2)], fill="black")

        if label:
            lx, ly = label_pos if label_pos else ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - 10)
            bbox = draw.textbbox((lx, ly), label, font=font_small, anchor="mm")
            # white background behind text
            draw.rectangle([bbox[0]-2, bbox[1]-1, bbox[2]+2, bbox[3]+1], fill="white")
            draw.text((lx, ly), label, fill="black", font=font_small, anchor="mm")

    # 1. Dataset
    draw_box(50, 70, 115, 55, ["Dataset", "IEMOCAP and", "Librispeech"])

    # 2. Audio Wave
    draw_box(50, 175, 115, 55, ["Audio Wave"])

    # 3. Input Transcript
    draw_box(50, 290, 115, 55, ["Input Transcript"])

    # 4. Conformer Student
    draw_box(235, 170, 135, 68, ["Conformer based", "Encoder- Decoder", "Model", "(Student)"])

    # 5. Dialogue Segmentation
    draw_box(235, 290, 135, 55, ["1) Dialogue", "Segmentation", "2) Human Annotation"])

    # 6. LoRA Adaptation LLaMA (rounded)
    draw_box(430, 290, 130, 55, ["LoRA Adaptation", "Llama 3.2 1B", "Model"], rounded=True, radius=10)

    # 7. Pseudo Bidirectional
    draw_box(430, 400, 130, 55, ["Pseudo Bidirectional", "Approach", "sequence length L' =8"])

    # 8. Sinkhorn OT Plan
    draw_box(620, 135, 150, 115, [
        "Sinkhorn Based Optimal",
        "Transport Plan",
        "Many to Many Token",
        "Mapping",
        "1) Ground Cost Matrix",
        "2) Top -K candidates"
    ])

    # 9. ASR Result
    draw_box(835, 160, 115, 55, ["ASR Result", "predicted Token"])

    # 10. Emotion Head (Corrected addition)
    draw_box(835, 245, 115, 55, ["Speech Emotion", "Recognition Head", "(Temporal Pooling)"])

    # 11. Wasserstein Loss
    draw_box(615, 335, 160, 50, ["Wasserstein Optimal", "Transport Loss"])

    # 12. Total Loss L
    draw_box(835, 335, 115, 50, ["Total Loss L"])

    # 13. Evaluation Metrics
    draw_box(1015, 155, 130, 145, [
        "Evaluation Metrics",
        "",
        "1) CER",
        "2) WER",
        "3) Weighted Accuracy",
        "4) Unweighted",
        "   Accuracy",
        "5) CTC Loss"
    ], left_align=True)

    # 14. Output
    draw_box(1205, 180, 120, 95, [
        "Output",
        "",
        "1) Recognized",
        "   Speech",
        "2) Emotion Label"
    ], left_align=True)

    # Arrows
    draw_arrow((107, 125), (107, 175))  # Dataset -> Audio
    draw_arrow((107, 230), (107, 290))  # Audio -> Transcript
    draw_arrow((165, 202), (235, 202), label="Raw Audio", label_pos=(200, 195)) # Audio -> Conformer
    draw_arrow((165, 317), (235, 317), label="Raw Text", label_pos=(200, 310))  # Transcript -> Dialogue
    draw_arrow((370, 317), (430, 317))  # Dialogue -> LLaMA
    draw_arrow((495, 400), (495, 345))  # Bidirectional -> LLaMA

    # Conformer -> Sinkhorn
    draw_arrow((370, 192), (620, 192), label="Audio Embeddings", label_pos=(480, 185))

    # LLaMA -> Sinkhorn (orthogonal)
    draw.line([(560, 317), (590, 317), (590, 215), (620, 215)], fill="black", width=1)
    draw_arrow((590, 215), (620, 215))
    bbox = draw.textbbox((590, 260), "Text Embeddings", font=font_small, anchor="mm")
    draw.rectangle([bbox[0]-2, bbox[1]-1, bbox[2]+2, bbox[3]+1], fill="white")
    draw.text((590, 260), "Text Embeddings", fill="black", font=font_small, anchor="mm")

    # Conformer -> Emotion Head (orthogonal)
    draw.line([(370, 220), (400, 220), (400, 272), (835, 272)], fill="black", width=1)
    draw_arrow((400, 272), (835, 272))

    # Sinkhorn -> ASR Result
    draw_arrow((770, 187), (835, 187))

    # Sinkhorn -> Wasserstein Loss
    draw_arrow((695, 250), (695, 335))

    # ASR Result -> Total Loss
    draw_arrow((892, 215), (892, 335))

    # Emotion Head -> Total Loss
    draw_arrow((925, 300), (925, 335))

    # Wasserstein Loss -> Total Loss
    draw_arrow((775, 360), (835, 360))

    # Total Loss -> Conformer (Backprop loop)
    draw.line([(835, 375), (302, 375), (302, 238)], fill="black", width=1)
    draw_arrow((302, 260), (302, 238))
    bbox = draw.textbbox((560, 375), "Updation of ASR weights Only", font=font_small, anchor="mm")
    draw.rectangle([bbox[0]-2, bbox[1]-1, bbox[2]+2, bbox[3]+1], fill="white")
    draw.text((560, 375), "Updation of ASR weights Only", fill="black", font=font_small, anchor="mm")

    # Total Loss -> Evaluation Metrics
    draw.line([(950, 360), (1080, 360), (1080, 300)], fill="black", width=1)
    draw_arrow((1080, 310), (1080, 300))
    bbox = draw.textbbox((1015, 360), "Minimize Loss", font=font_small, anchor="mm")
    draw.rectangle([bbox[0]-2, bbox[1]-1, bbox[2]+2, bbox[3]+1], fill="white")
    draw.text((1015, 360), "Minimize Loss", fill="black", font=font_small, anchor="mm")

    # Metrics -> Output
    draw_arrow((1145, 227), (1205, 227))

    img.save(output_path, dpi=(300, 300))
    print(f"--> Diagram successfully generated at: {output_path}")

if __name__ == "__main__":
    create_template_image()
