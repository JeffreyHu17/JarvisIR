import os
import re
import random
import gradio as gr
import torch
import argparse
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, TextIteratorStreamer
from threading import Thread
from agent_tools import RestorationToolkit

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run JarvisIR Gradio demo')
parser.add_argument('--model_path', type=str, required=True, help='Path to the fine-tuned LLaVA model')
parser.add_argument('--cuda_device', type=str, default="0", help='CUDA device to use (default: 0)')
args = parser.parse_args()

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

# Model configuration
# Use model path from command line arguments
model_id = args.model_path
    
# Available image restoration tasks and their corresponding models
all_tasks = " {denoise: [scunet, restormer], lighten: [retinexformer_fivek, hvicidnet, lightdiff], \
                derain: [idt, turbo_rain, s2former], defog:[ridcp, kanet], \
                desnow:[turbo_snow, snowmaster], super_resolution: [real_esrgan], \
            }"

# Various prompt templates for querying the LLM about image degradation and restoration tasks
prompts_query2 = [
    f"Considering the image's degradation, suggest the required tasks with explanations, and identify suitable tools for each task. Options for tasks and tools include: {all_tasks}.",
    f"Given the image's degradation, outline the essential tasks along with justifications, and choose the appropriate tools for each task from the following options: {all_tasks}.",
    f"Please specify the tasks required due to the image's degradation, explain the reasons, and select relevant tools for each task from the provided options: {all_tasks}.",
    f"Based on the image degradation, determine the necessary tasks and their reasons, along with the appropriate tools for each task. Choose from these options: {all_tasks}.",
    f"Identify the tasks required to address the image's degradation, including the reasons for each, and select tools from the options: {all_tasks}.",
    f"Considering the degradation observed, list the tasks needed and their justifications, then pick the most suitable tools for each task from these options: {all_tasks}.",
    f"Evaluate the image degradation, and based on that, provide the necessary tasks and reasons, along with tools chosen from the options: {all_tasks}.",
    f"With respect to the image degradation, outline the tasks needed and explain why, selecting tools from the following list: {all_tasks}.",
    f"Given the level of degradation in the image, specify tasks to address it, include reasons, and select tools for each task from: {all_tasks}.",
    f"Examine the image's degradation, propose relevant tasks and their explanations, and identify tools from the options provided: {all_tasks}.",
    f"Based on observed degradation, detail the tasks required, explain your choices, and select tools from these options: {all_tasks}.",
    f"Using the image's degradation as a guide, list the necessary tasks, include explanations, and pick tools from the provided choices: {all_tasks}.",
    f"Assess the image degradation, provide the essential tasks and reasons, and select the appropriate tools for each task from the options: {all_tasks}.",
    f"According to the image's degradation, determine which tasks are necessary and why, choosing tools for each task from: {all_tasks}.",
    f"Observe the degradation in the image, specify the needed tasks with justifications, and select appropriate tools from: {all_tasks}.",
    f"Taking the image degradation into account, specify tasks needed, provide reasons, and choose tools from the following: {all_tasks}.",
    f"Consider the image's degradation level, outline the tasks necessary, provide reasoning, and select suitable tools from: {all_tasks}.",
    f"Evaluate the degradation in the image, identify tasks required, explain your choices, and pick tools from: {all_tasks}.",
    f"Analyze the image degradation and suggest tasks with justifications, choosing the best tools from these options: {all_tasks}.",
    f"Review the image degradation, and based on it, specify tasks needed, provide reasons, and select tools for each task from: {all_tasks}."
]

# Initialize models
print("Loading LLM model...")

# Initialize the image restoration toolkit
tool_engine = RestorationToolkit(score_weight=[0,0,0,0,0])
# Load the LLaVA model in half precision to reduce memory usage
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to(0)
processor = AutoProcessor.from_pretrained(model_id)

print("Loading tool engine...")

def parse_llm_response(response):
    """
    Parse the LLM response to extract reason and answer sections
    
    Args:
        response (str): The raw response from the LLM
        
    Returns:
        tuple: (reason, answer) extracted from the response
    """
    reason_match = re.search(r'<reason>(.*?)</reason>', response, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    
    reason = reason_match.group(1).strip() if reason_match else "No reasoning provided"
    answer = answer_match.group(1).strip() if answer_match else "No answer provided"
    
    return reason, answer

def extract_models_from_answer(answer):
    """
    Extract model names from the answer string using regex
    
    Args:
        answer (str): The answer string containing model recommendations
        
    Returns:
        list: List of extracted model names
    """
    # Pattern to match [type:xxx]:(model:xxx)
    pattern = r'\[type:[^\]]+\]:\(model:([^)]+)\)'
    models = re.findall(pattern, answer)
    return models

def beautify_recommended_actions(answer, models):
    """
    Format the LLM's recommendations in a more visually appealing way
    
    Args:
        answer (str): The raw answer from LLM
        models (list): List of extracted model names
        
    Returns:
        str: Beautified display of recommendations
    """
    
    # Task type to emoji mapping for visual enhancement
    task_icons = {
        'denoise': '🧹',
        'lighten': '💡', 
        'derain': '🌧️',
        'defog': '🌫️',
        'desnow': '❄️',
        'super_resolution': '🔍'
    }
    
    # Parse the answer to extract tasks and models
    pattern = r'\[type:([^\]]+)\]:\(model:([^)]+)\)'
    matches = re.findall(pattern, answer)
    
    if not matches:
        return f"**🎯 Recommended Actions:**\n\n{answer}\n\n**Extracted Models:** {', '.join(models) if models else 'None'}"
    
    # Create beautified display
    beautified = "**🎯 Recommended Actions:**\n"
    beautified += "> "
    
    # Create horizontal flow of actions
    action_parts = []
    for task_type, model_name in matches:
        task_type = task_type.strip()
        model_name = model_name.strip()
        
        # Get icon for task type
        icon = task_icons.get(task_type, '🔧')
        
        # Format task name (capitalize and replace underscores)
        task_display = task_type.title().replace('_', ' ')
        
        # Create action part: icon + task + model
        action_part = f"{icon} {task_display}：`{model_name}`"
        action_parts.append(action_part)
    
    # Join with arrows to show sequence
    beautified += " ➡ ".join(action_parts) + "\n\n"
    
    # Add summary information
    beautified += f"**📋 Processing Pipeline:** {len(matches)} steps\n"
    beautified += f"**🛠️ Models to use:** {' → '.join(models)}"
    
    return beautified

def resize_image_to_original(processed_image_path, original_size):
    """
    Resize processed image back to original dimensions
    
    Args:
        processed_image_path (str): Path to the processed image
        original_size (tuple): Original image dimensions (width, height)
        
    Returns:
        str: Path to the resized image
    """
    if processed_image_path and os.path.exists(processed_image_path):
        img = Image.open(processed_image_path)
        img_resized = img.resize(original_size, Image.Resampling.LANCZOS)
        
        # Save resized image
        output_path = os.path.join('temp_outputs', 'final_result.png')
        img_resized.save(output_path)
        return output_path
    return processed_image_path

def get_llm_response_streaming(image_path):
    """
    Get streaming response from LLM for image analysis
    
    Args:
        image_path (str): Path to the input image
        
    Returns:
        TextIteratorStreamer: A streamer object to yield tokens
    """
    # Select random prompt from the templates
    instruction = prompts_query2[random.randint(0, len(prompts_query2)-1)]
    
    # Format the prompt with image for multimodal input
    prompt = (f"<|start_header_id|>user<|end_header_id|>\n\n<image>\n{instruction}<|eot_id|>"
              "<|start_header_id|>assistant<|end_header_id|>\n\n")
    
    # Load and process image
    raw_image = Image.open(image_path)
    inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)
    
    # Setup streaming for token-by-token generation
    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Generate response in a separate thread to avoid blocking
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=400,
        do_sample=False
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    return streamer

def process_image_with_tools(image_path, models, original_size):
    """
    Process image using the tool engine and restore to original size
    
    Args:
        image_path (str): Path to the input image
        models (list): List of models to apply
        original_size (tuple): Original image dimensions
        
    Returns:
        str: Path to the final processed image
    """
    if not models:
        return None
    
    # Create output directory
    os.makedirs('temp_outputs', exist_ok=True)
    
    # Process the image with selected models
    res = tool_engine.process_image(models, image_path, 'temp_outputs')
    
    # Resize back to original dimensions
    final_result = resize_image_to_original(res['output_path'], original_size)
    
    return final_result

def process_full_pipeline(image):
    """
    Main processing pipeline with streaming UI updates
    
    Args:
        image (str): Path to the input image
        
    Yields:
        tuple: (chat_history, processed_image) for Gradio UI updates
    """
    if image is None:
        return [], None
    
    try:
        # Get original image size for later restoration
        original_img = Image.open(image)
        original_size = original_img.size
        
        # Initialize chat history for UI
        chat_history = [("Image uploaded for analysis", None)]
        
        # Step 1: Get streaming LLM response
        streamer = get_llm_response_streaming(image)
        
        # Stream the response to UI with real-time updates
        full_response = ""
        in_reason = False
        in_answer = False
        reason_displayed = False
        answer_displayed = False
        reasoning_added = False  # Track if reasoning entry was added
        
        for new_text in streamer:
            full_response += new_text
            
            # Check if we're entering reason section or if we need to start showing content
            if ('<reason>' in full_response and not in_reason and not reason_displayed) or (not reasoning_added and not in_reason and not reason_displayed):
                in_reason = True
                reasoning_added = True
                
                if '<reason>' in full_response:
                    # Extract content after <reason>
                    reason_start = full_response.find('<reason>') + len('<reason>')
                    reason_content = full_response[reason_start:].strip()
                else:
                    # Show all content as reasoning if no tag yet
                    reason_content = full_response.strip()
                
                # Add reasoning to chat history
                chat_history.append((None, f"**🤔 Analysis & Reasoning:**\n\n{reason_content}"))
                yield chat_history, None
            
            # If we're in reason section, update content
            elif in_reason and not reason_displayed:
                # Check if reason section is complete
                if '</reason>' in full_response:
                    # Extract complete reason content
                    reason_start = full_response.find('<reason>') + len('<reason>')
                    reason_end = full_response.find('</reason>')
                    reason_content = full_response[reason_start:reason_end].strip()
                    
                    # Update chat history with complete reason
                    chat_history[1] = (None, f"**🤔 Analysis & Reasoning:**\n\n{reason_content}")
                    reason_displayed = True
                    in_reason = False
                    yield chat_history, None
                else:
                    # Continue streaming reason content
                    if '<reason>' in full_response:
                        reason_start = full_response.find('<reason>') + len('<reason>')
                        reason_content = full_response[reason_start:].strip()
                    else:
                        reason_content = full_response.strip()
                    
                    # Update chat history with partial reason
                    chat_history[1] = (None, f"**🤔 Analysis & Reasoning:**\n\n{reason_content}")
                    yield chat_history, None
            
            # Check if we're entering answer section
            elif '<answer>' in full_response and not in_answer and not answer_displayed and reason_displayed:
                in_answer = True
                # Extract content after <answer>
                answer_start = full_response.find('<answer>') + len('<answer>')
                answer_content = full_response[answer_start:]
                
                # Add partial answer to chat history
                models = extract_models_from_answer(answer_content)
                beautified = beautify_recommended_actions(answer_content, models)
                chat_history.append((None, beautified))
                yield chat_history, None
            
            # If we're in answer section, update content
            elif in_answer and not answer_displayed:
                # Check if answer section is complete
                if '</answer>' in full_response:
                    # Extract complete answer content
                    answer_start = full_response.find('<answer>') + len('<answer>')
                    answer_end = full_response.find('</answer>')
                    answer_content = full_response[answer_start:answer_end].strip()
                    
                    # Parse and process final answer
                    models = extract_models_from_answer(answer_content)
                    beautified = beautify_recommended_actions(answer_content, models)
                    chat_history[2] = (None, beautified)
                    answer_displayed = True
                    in_answer = False
                    yield chat_history, None
                    
                    # Process image with tools
                    if models:
                        chat_history.append((None, "**🔄 Processing image...**"))
                        yield chat_history, None
                        
                        processed_image = process_image_with_tools(image, models, original_size)
                        chat_history[-1] = (None, "**✅ Processing Complete!**")
                        yield chat_history, processed_image
                        return
                    else:
                        chat_history.append((None, "**❌ No valid models found in the response**"))
                        yield chat_history, None
                        return
                else:
                    # Continue streaming answer content
                    answer_start = full_response.find('<answer>') + len('<answer>')
                    answer_content = full_response[answer_start:].strip()
                    
                    # Update chat history with partial answer
                    models = extract_models_from_answer(answer_content)
                    beautified = beautify_recommended_actions(answer_content, models)
                    chat_history[2] = (None, beautified)
                    yield chat_history, None
        
        # Fallback if streaming completes without proper tags
        if not answer_displayed:
            reason, answer = parse_llm_response(full_response)
            models = extract_models_from_answer(answer)
            
            chat_history = [
                ("Image uploaded for analysis", None),
                (None, f"**🤔 Analysis & Reasoning:**\n\n{reason}"),
                (None, beautify_recommended_actions(answer, models))
            ]
            
            if models:
                chat_history.append((None, "**🔄 Processing image...**"))
                yield chat_history, None
                
                processed_image = process_image_with_tools(image, models, original_size)
                chat_history[-1] = (None, "**✅ Processing Complete!**")
                yield chat_history, processed_image
            else:
                chat_history.append((None, "**❌ No valid models found in the response**"))
                yield chat_history, None
            
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        chat_history = [
            ("Image uploaded for analysis", None),
            (None, f"**❌ Error occurred:**\n\n{error_msg}")
        ]
        yield chat_history, None

# Create Gradio interface
def create_interface():
    """
    Create and configure the Gradio web interface
    
    Returns:
        gr.Blocks: Configured Gradio interface
    """
    with gr.Blocks(title="JarvisIR: Elevating Autonomous Driving Perception with Intelligent Image Restoration", theme=gr.themes.Soft()) as demo:
        # Header with logo and title
        gr.Markdown("""
        # <img src="https://cvpr2025-jarvisir.github.io/imgs/icon.png" width="32" height="32" style="display: inline-block; vertical-align: middle; transform: translateY(-2px); margin-right: 1px;"/> JarvisIR: Elevating Autonomous Driving Perception with Intelligent Image Restoration
        
        Upload an image and let JarvisIR analyze its degradation and recommend the best restoration tools!
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input image upload component
                input_image = gr.Image(
                    type="filepath", 
                    label="📸 Upload Your Image",
                    height=400
                )
                
                # Process button
                process_btn = gr.Button(
                    "🚀 Analyze & Process", 
                    variant="primary",
                    size="lg"
                )
                
            with gr.Column(scale=1):
                # Chat interface to show analysis
                chatbot = gr.Chatbot(
                    label="💬 AI Analysis Chat",
                    height=400,
                    show_label=True,
                    bubble_full_width=False
                )
        
        with gr.Row():
            # Output image display
            output_image = gr.Image(
                label="✨ Processed Result", 
                height=300
            )
        
        # Connect event handler for the process button
        process_btn.click(
            fn=process_full_pipeline,
            inputs=[input_image],
            outputs=[chatbot, output_image]
        )
        
        # Instructions section
        gr.Markdown("### 📝 Instructions:")
        gr.Markdown("""
        1. **Upload an image** that needs restoration (blurry, dark, noisy, etc.)
        2. **Click 'Analyze & Process'** to let AI analyze the image
        3. **View the chat** to see AI's reasoning and recommendations in real-time
        4. **Check the result** - processed image restored to original dimensions
        """)
    
    return demo

if __name__ == "__main__":
    print("Starting Image Restoration Assistant...")
    demo = create_interface()
    # Launch the Gradio app on specified host and port
    demo.launch(
        server_name="0.0.0.0",
        server_port=7866,
        share=False
    )
