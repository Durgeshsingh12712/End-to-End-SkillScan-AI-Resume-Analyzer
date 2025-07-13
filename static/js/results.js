// Animate confidence meter on load
        document.addEventListener('DOMContentLoaded', function() {
            const confidenceFill = document.querySelector('.confidence-fill');
            if (confidenceFill) {
                const width = confidenceFill.style.width;
                confidenceFill.style.width = '0%';
                setTimeout(() => {
                    confidenceFill.style.width = width;
                }, 500);
            }
        });

        // Copy results to clipboard
        function copyResults() {
            const results = {
                filename: "{{ results.filename }}",
                category: "{{ results.prediction.prediction }}",
                confidence: "{{ '%.1f'|format(results.prediction.confidence * 100) }}%",
                skills_found: {{ results.skill_analysis.skill_count }},
                skills: {{ results.skill_analysis.found_skills|tojson }},
                suggestions: {{ results.skill_analysis.improvement_suggestions|tojson }},
                analysis_time: "{{ results.upload_time }}"
            };

            const resultText = `Resume Analysis Results
=====================================
File: ${results.filename}
Category: ${results.category}
Confidence: ${results.confidence}
Skills Found: ${results.skills_found}

Detected Skills:
${results.skills.join(', ')}

Improvement Suggestions:
${results.suggestions.map((s, i) => `${i + 1}. ${s}`).join('\n')}

Analysis Date: ${results.analysis_time}
=====================================`;

            navigator.clipboard.writeText(resultText).then(() => {
                // Show success message
                const btn = event.target;
                const originalText = btn.textContent;
                btn.textContent = '✅ Copied!';
                btn.style.background = '#10b981';
                
                setTimeout(() => {
                    btn.textContent = originalText;
                    btn.style.background = '';
                }, 2000);
            }).catch(err => {
                console.error('Failed to copy: ', err);
                alert('Failed to copy results. Please try again.');
            });
        }

        // Add smooth scrolling
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                document.querySelector(this.getAttribute('href')).scrollIntoView({
                    behavior: 'smooth'
                });
            });
        });