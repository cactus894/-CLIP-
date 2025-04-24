import streamlit as st  # 用于构建前端UI界面
import requests  # 用于向后端 FastAPI 发送 HTTP 请求
import tempfile  # 用于创建临时文件，存储上传的视频
import time  # 计算视频处理时间


# 配置常量
API_ENDPOINT = "http://localhost:8000/api/v1/process"  # 后端 API 地址，用于处理视频
MAX_FILE_SIZE = 500  # 视频最大支持 500MB，超过限制则报错


def format_size(size):
    """将字节转换为MB，便于显示文件大小"""
    return size / (1024 * 1024)


def main():
    st.set_page_config(page_title="视频智能剪辑系统", layout="wide")  # 设置页面标题和布局，wide代表宽屏布局

    st.title("🎥 视频素材提取系统")
    st.markdown("---")

    # 文件上传，返回None或一个UploadedFile对象，如果accept_multiple_files=True则返回列表
    video_file = st.file_uploader(
        "上传视频文件",
        type=["mp4", "mov", "mkv"],
        help=f"最大支持{MAX_FILE_SIZE}MB文件"
    )

    # 显示预览
    if video_file:
        if format_size(video_file.size) > MAX_FILE_SIZE:
            st.error(f"文件大小超过限制 ({MAX_FILE_SIZE}MB)")
            return

        # 创建一个临时文件，delete=False 让文件在程序运行结束前不会被删除
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        st.video(tfile.name)  # 播放视频

    # 输入区域
    with st.form("input_form"):  # 将多个输入控件组合在一起
        text_query = st.text_input(
            "素材描述",
            placeholder="示例：会议室中正在演讲的男士特写镜头"
        )

        col1, col2 = st.columns(2)
        with col1:
            min_duration = st.slider("最小片段时长(秒)", 1, 10, 3)
        with col2:
            threshold = st.slider("匹配阈值", 0.0, 1.0, 0.3)

        submitted = st.form_submit_button("开始提取")

    # 处理逻辑
    if submitted and video_file and text_query:
        with st.spinner("视频分析中，请稍候..."):
            try:
                start_time = time.time()

                # 向API_ENDPOINT发送一个HTTP POST请求，使用with语句确保文件正确关闭
                with open(tfile.name, "rb") as video_file:
                    response = requests.post(
                        API_ENDPOINT,
                        files={"video": video_file},
                        data={
                            "text": text_query,
                            "min_duration": min_duration,
                            "threshold": threshold
                        },

                        # 设定超时时间为300秒
                        timeout=300
                    )

                # 处理结果
                if response.status_code == 200:
                    output_path = f"./frontend/assets/result_{int(time.time())}.mp4"
                    with open(output_path, "wb") as f:
                        # 将后端返回的视频数据写入本地文件
                        f.write(response.content)

                    # 在Streamlit界面上显示成功消息
                    st.success(f"处理完成！耗时 {time.time() - start_time:.1f}秒")
                    st.video(output_path)

                    # 下载按钮
                    st.download_button(
                        "下载结果视频",
                        data=response.content,
                        file_name="extracted_clip.mp4",
                        mime="video/mp4"
                    )
                else:
                    st.error(f"处理失败: {response.text}")

            except Exception as e:
                st.error(f"发生错误: {str(e)}")


if __name__ == "__main__":
    main()
