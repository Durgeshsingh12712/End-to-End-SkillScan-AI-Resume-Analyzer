const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const uploadBtn = document.getElementById('uploadBtn');
        const selectedFile = document.getElementById('selectedFile');
        const uploadForm = document.getElementById('uploadForm');
        const loadingSection = document.getElementById('loadingSection');
        const progressFill = document.getElementById('progressFill');

        // Click to upload
        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });

        // Drag and drop functionality
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                handleFileSelect(files[0]);
            }
        });

        // File selection handler
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileSelect(e.target.files[0]);
            }
        });

        function handleFileSelect(file) {
            const allowedTypes = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
            const maxSize = 2 * 1024 * 1024; // 2MB

            if (!allowedTypes.includes(file.type)) {
                alert('Please select a PDF or DOCX file');
                return;
            }

            if (file.size > maxSize) {
                alert('File size must be less than 2MB');
                return;
            }

            // Show selected file info
            selectedFile.querySelector('.filename').textContent = file.name;
            selectedFile.querySelector('.filesize').textContent = `${(file.size / 1024).toFixed(1)} KB`;
            selectedFile.style.display = 'block';
            
            // Enable upload button
            uploadBtn.disabled = false;
            uploadBtn.textContent = 'Analyze Resume';
        }

        // Form submission with progress
        uploadForm.addEventListener('submit', (e) => {
            e.preventDefault();
            
            if (!fileInput.files.length) {
                alert('Please select a file first');
                return;
            }

            // Show loading state
            uploadForm.style.display = 'none';
            loadingSection.style.display = 'block';
            
            // Simulate progress
            let progress = 0;
            const progressInterval = setInterval(() => {
                progress += Math.random() * 15;
                if (progress > 90) progress = 90;
                progressFill.style.width = progress + '%';
            }, 200);

            // Submit form
            const formData = new FormData(uploadForm);
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                clearInterval(progressInterval);
                progressFill.style.width = '100%';
                
                if (response.ok) {
                    return response.text();
                } else {
                    throw new Error('Upload failed');
                }
            })
            .then(html => {
                // Replace page content with results
                document.body.innerHTML = html;
            })
            .catch(error => {
                clearInterval(progressInterval);
                console.error('Error:', error);
                alert('Upload failed. Please try again.');
                
                // Reset form
                uploadForm.style.display = 'block';
                loadingSection.style.display = 'none';
                progressFill.style.width = '0%';
            });
        });

        // File input validation
        fileInput.addEventListener('change', function() {
            const file = this.files[0];
            if (file) {
                const fileExtension = file.name.split('.').pop().toLowerCase();
                if (!['pdf', 'docx'].includes(fileExtension)) {
                    alert('Please select a PDF or DOCX file only.');
                    this.value = '';
                    uploadBtn.disabled = true;
                    selectedFile.style.display = 'none';
                }
            }
        });