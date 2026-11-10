/**
 * Brain MRI Image Restoration Dashboard - Frontend Interaction
 */

document.addEventListener('DOMContentLoaded', () => {
    // 1. Mobile Sidebar Toggle
    const sidebarToggle = document.getElementById('sidebarToggle');
    const sidebar = document.querySelector('.app-sidebar');
    if (sidebarToggle && sidebar) {
        sidebarToggle.addEventListener('click', () => {
            sidebar.classList.toggle('show');
        });
    }

    // 2. Drag & Drop File Upload Handler
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('imageFileInput');
    const previewContainer = document.getElementById('uploadPreviewContainer');
    const previewImage = document.getElementById('uploadPreviewImage');
    const uploadForm = document.getElementById('imageUploadForm');
    const uploadSubmitBtn = document.getElementById('uploadSubmitBtn');

    if (dropzone && fileInput) {
        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropzone.classList.add('dragover');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropzone.classList.remove('dragover');
            }, false);
        });

        dropzone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                fileInput.files = files;
                handleFileSelect(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileSelect(e.target.files[0]);
            }
        });
    }

    function handleFileSelect(file) {
        // Validate MIME type
        const validTypes = ['image/png', 'image/jpeg', 'image/jpg'];
        if (!validTypes.includes(file.type)) {
            alert('Invalid file format. Please upload a PNG or JPEG MRI slice.');
            fileInput.value = '';
            return;
        }

        // Validate File Size (Max 10 MB)
        if (file.size > 10 * 1024 * 1024) {
            alert('File exceeds 10 MB limit. Please select a standard MRI slice.');
            fileInput.value = '';
            return;
        }

        // Show Local Preview
        const reader = new FileReader();
        reader.onload = (e) => {
            if (previewImage && previewContainer) {
                previewImage.src = e.target.result;
                previewContainer.classList.remove('d-none');
            }
            if (uploadSubmitBtn) {
                uploadSubmitBtn.disabled = false;
            }
        };
        reader.readAsDataURL(file);
    }

    // 3. Severity Slider Value Update
    const severitySlider = document.getElementById('severitySlider');
    const severityDisplay = document.getElementById('severityDisplay');
    if (severitySlider && severityDisplay) {
        severitySlider.addEventListener('input', (e) => {
            severityDisplay.textContent = `${e.target.value}%`;
        });
    }
});
