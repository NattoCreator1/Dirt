#!/usr/bin/env python3
"""
合成图像筛选工具

功能:
- Web界面浏览生成的合成图像
- 支持Accept/Reject标记
- 显示质量指标(spill_rate, mask_coverage等)
- 支持键盘快捷键操作
- 导出筛选结果

使用方法:
    python filter_synthetic_images.py --data_dir /path/to/batch_896f_10masks

Author: SD Experiment Team
Date: 2026-02-18
"""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template_string, jsonify, request, send_file
import pandas as pd

# ============================================================================
# HTML模板
# ============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>合成图像筛选工具</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
        }
        .container {
            display: flex;
            height: 100vh;
        }
        /* 侧边栏 */
        .sidebar {
            width: 300px;
            background: #16213e;
            padding: 20px;
            overflow-y: auto;
            border-right: 1px solid #0f3460;
        }
        .sidebar h2 {
            font-size: 18px;
            margin-bottom: 20px;
            color: #e94560;
        }
        .stats {
            background: #0f3460;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .stats-item {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 14px;
        }
        .stats-value {
            font-weight: bold;
            color: #4cc9f0;
        }
        .progress-bar {
            width: 100%;
            height: 20px;
            background: #0f3460;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4cc9f0, #4361ee);
            transition: width 0.3s;
        }
        .filters {
            margin-bottom: 20px;
        }
        .filter-group {
            margin-bottom: 15px;
        }
        .filter-group label {
            display: block;
            margin-bottom: 5px;
            font-size: 12px;
            color: #888;
        }
        .filter-group select, .filter-group input {
            width: 100%;
            padding: 8px;
            border-radius: 4px;
            border: 1px solid #0f3460;
            background: #0a0a1a;
            color: #eee;
        }
        .btn {
            width: 100%;
            padding: 10px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            margin-bottom: 10px;
            transition: transform 0.1s;
        }
        .btn:hover { transform: scale(1.02); }
        .btn:active { transform: scale(0.98); }
        .btn-primary { background: #4361ee; color: white; }
        .btn-success { background: #06d6a0; color: white; }
        .btn-danger { background: #ef476f; color: white; }
        .btn-secondary { background: #6c757d; color: white; }
        /* 主区域 */
        .main {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        .toolbar {
            padding: 15px 20px;
            background: #16213e;
            display: flex;
            gap: 10px;
            align-items: center;
            border-bottom: 1px solid #0f3460;
        }
        .toolbar input {
            padding: 8px 12px;
            border-radius: 4px;
            border: 1px solid #0f3460;
            background: #0a0a1a;
            color: #eee;
            width: 300px;
        }
        .image-grid {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
            align-content: start;
        }
        .image-card {
            background: #16213e;
            border-radius: 8px;
            overflow: hidden;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            border: 2px solid transparent;
        }
        .image-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }
        .image-card.accepted {
            border-color: #06d6a0;
        }
        .image-card.rejected {
            border-color: #ef476f;
            opacity: 0.6;
        }
        .image-card.viewed {
            border-color: #ffd166;
        }
        .image-card img {
            width: 100%;
            aspect-ratio: 1;
            object-fit: cover;
        }
        .image-info {
            padding: 10px;
            font-size: 11px;
        }
        .image-info .filename {
            font-weight: bold;
            margin-bottom: 5px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .image-info .metrics {
            display: flex;
            justify-content: space-between;
            color: #888;
        }
        .image-info .metrics span.good {
            color: #06d6a0;
        }
        .image-info .metrics span.bad {
            color: #ef476f;
        }
        /* 模态框 */
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.9);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }
        .modal.active { display: flex; }
        .modal-content {
            max-width: 95vw;
            max-height: 90vh;
            display: flex;
            gap: 20px;
            transition: all 0.3s;
        }
        /* 单图模式 */
        .modal-content.single-mode {
            flex-direction: row;
        }
        /* 对比模式 */
        .modal-content.compare-mode {
            flex-direction: row;
            align-items: stretch;
        }
        .modal-content.compare-mode .image-container {
            display: flex;
            gap: 10px;
            flex: 1;
            min-width: 0;
        }
        .modal-content.compare-mode .image-wrapper {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            background: #16213e;
            border-radius: 8px;
            padding: 10px;
        }
        .modal-content.compare-mode .image-wrapper img {
            max-height: 70vh;
            max-width: 100%;
            object-fit: contain;
        }
        .modal-content.compare-mode .image-wrapper .image-label {
            margin-top: 10px;
            font-size: 14px;
            font-weight: bold;
            color: #4cc9f0;
        }
        .modal-image {
            max-height: 85vh;
            max-width: 70vw;
            object-fit: contain;
        }
        .image-container {
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .modal-info {
            width: 300px;
            background: #16213e;
            padding: 20px;
            border-radius: 8px;
            overflow-y: auto;
        }
        .modal-info h3 {
            margin-bottom: 15px;
            color: #e94560;
        }
        .modal-info .metric {
            margin-bottom: 10px;
            padding: 8px;
            background: #0f3460;
            border-radius: 4px;
            font-size: 13px;
        }
        .modal-info .metric-label {
            color: #888;
            margin-bottom: 3px;
        }
        .modal-info .metric-value {
            font-size: 16px;
            font-weight: bold;
        }
        .modal-actions {
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }
        .modal-actions .btn {
            margin-bottom: 0;
            padding: 15px;
            font-size: 16px;
        }
        .keyboard-hint {
            margin-top: 15px;
            padding: 10px;
            background: #0f3460;
            border-radius: 4px;
            font-size: 11px;
            color: #888;
        }
        .keyboard-hint kbd {
            background: #eee;
            color: #333;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: monospace;
        }
        /* 空状态 */
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #888;
        }
        .empty-state h3 {
            font-size: 24px;
            margin-bottom: 10px;
        }
        /* 导航箭头 */
        .nav-arrow {
            position: fixed;
            top: 50%;
            transform: translateY(-50%);
            background: rgba(22, 33, 62, 0.8);
            color: #eee;
            border: none;
            font-size: 30px;
            padding: 20px 15px;
            cursor: pointer;
            z-index: 999;
            border-radius: 8px;
        }
        .nav-arrow:hover { background: rgba(67, 97, 238, 0.8); }
        .nav-prev { left: 20px; }
        .nav-next { right: 20px; }
        .close-modal {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(239, 71, 111, 0.8);
            color: white;
            border: none;
            font-size: 24px;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            cursor: pointer;
            z-index: 1001;
        }
        .close-modal:hover { background: #ef476f; }
    </style>
</head>
<body>
    <div class="container">
        <!-- 侧边栏 -->
        <div class="sidebar">
            <h2>🔍 图像筛选工具</h2>

            <div class="stats">
                <div class="stats-item">
                    <span>总图像</span>
                    <span class="stats-value" id="total-count">-</span>
                </div>
                <div class="stats-item">
                    <span>已浏览</span>
                    <span class="stats-value" id="viewed-count">-</span>
                </div>
                <div class="stats-item">
                    <span>已接受</span>
                    <span class="stats-value" style="color:#06d6a0" id="accepted-count">-</span>
                </div>
                <div class="stats-item">
                    <span>已拒绝</span>
                    <span class="stats-value" style="color:#ef476f" id="rejected-count">-</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="progress-fill"></div>
                </div>
                <div class="stats-item">
                    <span>进度</span>
                    <span class="stats-value" id="progress-text">0%</span>
                </div>
            </div>

            <div class="filters">
                <div class="filter-group">
                    <label>筛选状态</label>
                    <select id="status-filter">
                        <option value="all">全部</option>
                        <option value="pending">待浏览</option>
                        <option value="accepted">已接受 ✓</option>
                        <option value="rejected">已拒绝 ✗</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>Spill Rate ≤</label>
                    <input type="range" id="spill-filter" min="0" max="100" value="100">
                    <span id="spill-value">100%</span>
                </div>
                <div class="filter-group">
                    <label>排序</label>
                    <select id="sort-by">
                        <option value="filename">文件名</option>
                        <option value="spill_asc">Spill Rate (低→高)</option>
                        <option value="spill_desc">Spill Rate (高→低)</option>
                    </select>
                </div>
            </div>

            <button class="btn btn-primary" onclick="loadNextBatch()">📥 加载更多</button>
            <button class="btn btn-success" onclick="exportResults()">💾 导出结果</button>
            <button class="btn btn-secondary" onclick="resetFilters()">🔄 重置筛选</button>

            <div class="filter-group" style="margin-top: 20px;">
                <label>跳转到</label>
                <input type="number" id="jump-to" placeholder="图像编号" min="1">
                <button class="btn btn-secondary" onclick="jumpToImage()" style="margin-top:5px">跳转</button>
            </div>
        </div>

        <!-- 主区域 -->
        <div class="main">
            <div class="toolbar">
                <input type="text" id="search-input" placeholder="搜索文件名..." onkeyup="filterImages()">
                <span id="showing-info" style="margin-left:auto; color:#888"></span>
            </div>
            <div class="image-grid" id="image-grid"></div>
        </div>
    </div>

    <!-- 模态框 -->
    <div class="modal" id="modal">
        <button class="close-modal" onclick="closeModal()">×</button>
        <button class="nav-arrow nav-prev" onclick="navigateImage(-1)">‹</button>
        <button class="nav-arrow nav-next" onclick="navigateImage(1)">›</button>

        <div class="modal-content" id="modal-content">
            <!-- 单图模式 -->
            <div class="image-container" id="single-image-view">
                <img class="modal-image" id="modal-image" src="" alt="">
            </div>

            <!-- 对比模式 -->
            <div class="image-container" id="compare-view" style="display:none;">
                <div class="image-wrapper">
                    <img id="clean-image" src="" alt="原始干净帧">
                    <div class="image-label">原始干净帧</div>
                </div>
                <div class="image-wrapper">
                    <img id="dirty-image" src="" alt="生成脏污图像">
                    <div class="image-label">生成脏污图像</div>
                </div>
            </div>

            <div class="modal-info">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
                    <h3 id="modal-filename" style="margin:0;">filename.png</h3>
                    <button class="btn btn-secondary" onclick="toggleCompareMode()" style="width:auto; padding:8px 15px; margin:0; font-size:12px;">
                        🔁 对比模式
                    </button>
                </div>

                <div class="metric">
                    <div class="metric-label">Spill Rate</div>
                    <div class="metric-value" id="modal-spill">-</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Mask Coverage</div>
                    <div class="metric-value" id="modal-coverage">-</div>
                </div>
                <div class="metric">
                    <div class="metric-label">类别</div>
                    <div class="metric-value" id="modal-class">-</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Caption</div>
                    <div class="metric-value" style="font-size:12px; font-weight:normal" id="modal-caption">-</div>
                </div>

                <div class="modal-actions">
                    <button class="btn btn-success" onclick="acceptCurrent()">✓ 接受</button>
                    <button class="btn btn-danger" onclick="rejectCurrent()">✗ 拒绝</button>
                </div>

                <div class="keyboard-hint">
                    <strong>快捷键:</strong><br>
                    <kbd>A</kbd> 接受 | <kbd>R</kbd> 拒绝<br>
                    <kbd>←</kbd> <kbd>→</kbd> 切换图像<br>
                    <kbd>C</kbd> 切换对比模式<br>
                    <kbd>Esc</kbd> 关闭
                </div>
            </div>
        </div>
    </div>

    <script>
        var images = [];
        var filteredImages = [];
        var decisions = {};
        var currentIndex = 0;
        var currentBatch = 0;
        var compareMode = false;
        var batchSize = 100;

        var API_URL = '/api';

        function init() {
            var xhr = new XMLHttpRequest();
            xhr.open('GET', API_URL + '/images', true);
            xhr.onload = function() {
                if (xhr.status === 200) {
                    try {
                        images = JSON.parse(xhr.responseText);
                        console.log('Loaded', images.length, 'images');
                        filteredImages = images.slice();
                        loadDecisions();
                        applyFilters();
                        updateStats();
                    } catch (error) {
                        console.error('Failed to parse images:', error);
                        document.getElementById('image-grid').innerHTML =
                            '<div class="empty-state"><h3>❌ 解析失败</h3><p>' + error.message + '</p></div>';
                    }
                } else {
                    document.getElementById('image-grid').innerHTML =
                        '<div class="empty-state"><h3>❌ 加载失败</h3><p>HTTP ' + xhr.status + '</p></div>';
                }
            };
            xhr.onerror = function() {
                document.getElementById('image-grid').innerHTML =
                    '<div class="empty-state"><h3>❌ 网络错误</h3><p>无法连接到服务器</p></div>';
            };
            xhr.send();
        }

        function loadDecisions() {
            var saved = localStorage.getItem('filter_decisions');
            if (saved) {
                decisions = JSON.parse(saved);
            }
        }

        function saveDecision(filename, status) {
            decisions[filename] = status;
            localStorage.setItem('filter_decisions', JSON.stringify(decisions));
            updateStats();
            updateCardStyle(filename);
        }

        function updateStats() {
            var accepted = 0;
            var rejected = 0;
            for (var key in decisions) {
                if (decisions[key] === 'accepted') accepted++;
                else if (decisions[key] === 'rejected') rejected++;
            }
            var viewed = accepted + rejected;

            document.getElementById('total-count').textContent = images.length;
            document.getElementById('viewed-count').textContent = viewed;
            document.getElementById('accepted-count').textContent = accepted;
            document.getElementById('rejected-count').textContent = rejected;

            var progress = (viewed / images.length * 100).toFixed(1);
            document.getElementById('progress-fill').style.width = progress + '%';
            document.getElementById('progress-text').textContent = progress + '%';
        }

        function applyFilters() {
            var statusFilter = document.getElementById('status-filter').value;
            var spillFilter = document.getElementById('spill-filter').value / 100;
            var sortBy = document.getElementById('sort-by').value;
            var searchText = document.getElementById('search-input').value.toLowerCase();

            filteredImages = images.filter(function(img) {
                // 状态筛选
                var status = decisions[img.filename];
                if (statusFilter === 'accepted' && status !== 'accepted') return false;
                if (statusFilter === 'rejected' && status !== 'rejected') return false;
                if (statusFilter === 'pending' && status) return false;

                // Spill rate筛选
                if (img.spill_rate > spillFilter) return false;

                // 搜索筛选
                if (searchText && img.filename.toLowerCase().indexOf(searchText) === -1) return false;

                return true;
            });

            console.log('Filtered:', filteredImages.length, 'images from', images.length);

            // 排序
            filteredImages.sort(function(a, b) {
                if (sortBy === 'filename') return a.filename.localeCompare(b.filename);
                if (sortBy === 'spill_asc') return a.spill_rate - b.spill_rate;
                if (sortBy === 'spill_desc') return b.spill_rate - a.spill_rate;
                return 0;
            });

            currentBatch = 0;
            renderGrid();
        }

        function renderGrid() {
            var grid = document.getElementById('image-grid');
            var start = currentBatch * batchSize;
            var end = start + batchSize;
            var batch = filteredImages.slice(start, end);

            if (batch.length === 0 && currentBatch === 0) {
                grid.innerHTML = '<div class="empty-state"><h3>😕 没有找到匹配的图像</h3><p>请调整筛选条件</p></div>';
                document.getElementById('showing-info').textContent = '';
                return;
            }

            grid.innerHTML = batch.map(function(img, idx) {
                var status = decisions[img.filename];
                var statusClass = status === 'accepted' ? 'accepted' :
                                  status === 'rejected' ? 'rejected' :
                                  status ? 'viewed' : '';
                var spillClass = img.spill_rate < 0.3 ? 'good' : img.spill_rate > 0.5 ? 'bad' : '';

                return '<div class="image-card ' + statusClass + '" data-filename="' + img.filename + '"' +
                       ' onclick="openModal(' + (start + idx) + ')">' +
                       '<img src="/images/' + img.filename + '" alt="' + img.filename + '" loading="lazy">' +
                       '<div class="image-info">' +
                       '<div class="filename">' + img.filename + '</div>' +
                       '<div class="metrics">' +
                       '<span class="' + spillClass + '">SR: ' + (img.spill_rate*100).toFixed(0) + '%</span>' +
                       '<span>C' + img.dominant_class + '</span>' +
                       '</div></div></div>';
            }).join('');

            document.getElementById('showing-info').textContent =
                '显示 ' + (start + 1) + '-' + Math.min(end, filteredImages.length) + ' / 共 ' + filteredImages.length + ' 张';
        }

        function updateCardStyle(filename) {
            var card = document.querySelector('[data-filename="' + filename + '"]');
            if (!card) return;

            var status = decisions[filename];
            card.classList.remove('accepted', 'rejected', 'viewed');
            if (status === 'accepted') card.classList.add('accepted');
            else if (status === 'rejected') card.classList.add('rejected');
            else if (status) card.classList.add('viewed');
        }

        function loadNextBatch() {
            currentBatch++;
            if ((currentBatch * batchSize) >= filteredImages.length) {
                currentBatch = 0;
            }
            renderGrid();
            window.scrollTo(0, 0);
        }

        function openModal(idx) {
            currentIndex = idx;
            var img = filteredImages[idx];
            if (!img) return;

            // 更新图像
            document.getElementById('modal-image').src = '/images/' + img.filename;
            document.getElementById('dirty-image').src = '/images/' + img.filename;

            // 更新干净帧
            if (img.clean_path) {
                document.getElementById('clean-image').src = '/clean/' + img.clean_path;
            } else {
                document.getElementById('clean-image').src = '';
            }

            document.getElementById('modal-filename').textContent = img.filename;

            var spillEl = document.getElementById('modal-spill');
            spillEl.textContent = (img.spill_rate * 100).toFixed(1) + '%';
            spillEl.style.color = img.spill_rate < 0.3 ? '#06d6a0' :
                                 img.spill_rate > 0.5 ? '#ef476f' : '#eee';

            document.getElementById('modal-coverage').textContent = (img.coverage * 100).toFixed(1) + '%';
            document.getElementById('modal-class').textContent = 'C' + img.dominant_class;
            document.getElementById('modal-caption').textContent = img.caption || '-';

            // 更新按钮状态
            var status = decisions[img.filename];
            var acceptBtn = document.querySelector('.modal-actions .btn-success');
            var rejectBtn = document.querySelector('.modal-actions .btn-danger');
            acceptBtn.style.opacity = status === 'accepted' ? '1' : '0.7';
            rejectBtn.style.opacity = status === 'rejected' ? '1' : '0.7';

            // 应用当前模式
            applyCompareMode();

            document.getElementById('modal').classList.add('active');
        }

        function closeModal() {
            document.getElementById('modal').classList.remove('active');
        }

        function toggleCompareMode() {
            compareMode = !compareMode;
            applyCompareMode();
        }

        function applyCompareMode() {
            var modalContent = document.getElementById('modal-content');
            var singleView = document.getElementById('single-image-view');
            var compareView = document.getElementById('compare-view');

            if (compareMode) {
                modalContent.classList.remove('single-mode');
                modalContent.classList.add('compare-mode');
                singleView.style.display = 'none';
                compareView.style.display = 'flex';
            } else {
                modalContent.classList.remove('compare-mode');
                modalContent.classList.add('single-mode');
                singleView.style.display = 'flex';
                compareView.style.display = 'none';
            }
        }

        function navigateImage(delta) {
            currentIndex = (currentIndex + delta + filteredImages.length) % filteredImages.length;
            openModal(currentIndex);
        }

        function acceptCurrent() {
            var img = filteredImages[currentIndex];
            saveDecision(img.filename, 'accepted');
            // 自动跳到下一张未浏览的
            navigateToNextPending();
        }

        function rejectCurrent() {
            var img = filteredImages[currentIndex];
            saveDecision(img.filename, 'rejected');
            // 自动跳到下一张未浏览的
            navigateToNextPending();
        }

        function navigateToNextPending() {
            // 查找下一张未浏览的图像
            for (var i = 1; i < filteredImages.length; i++) {
                var nextIdx = (currentIndex + i) % filteredImages.length;
                if (!decisions[filteredImages[nextIdx].filename]) {
                    openModal(nextIdx);
                    return;
                }
            }
            // 如果都浏览过了，关闭模态框
            closeModal();
        }

        function filterImages() {
            applyFilters();
        }

        function resetFilters() {
            document.getElementById('status-filter').value = 'all';
            document.getElementById('spill-filter').value = 100;
            document.getElementById('spill-value').textContent = '100%';
            document.getElementById('sort-by').value = 'filename';
            document.getElementById('search-input').value = '';
            applyFilters();
        }

        function exportResults() {
            // 前端直接生成CSV
            var timestamp = new Date().toISOString().slice(0,10).replace(/-/g, '');
            var filename = 'filtered_results_' + timestamp + '.csv';

            // 获取manifest数据用于额外信息
            var manifestData = {};
            var xhr = new XMLHttpRequest();
            xhr.open('GET', API_URL + '/manifest', false);
            try {
                xhr.send();
                if (xhr.status === 200) {
                    manifestData = JSON.parse(xhr.responseText);
                }
            } catch (e) {
                console.warn('无法获取manifest数据，使用基础导出');
            }

            // 构建CSV内容
            var headers = [
                'output_filename',
                'filter_status',
                'clean_path',
                'clean_file_id',
                'clean_camera',
                'mask_file_id',
                'spill_rate',
                'coverage',
                'mask_dominant_class',
                'caption'
            ];

            var rows = [];
            for (var imgFilename in decisions) {
                var status = decisions[imgFilename];
                var imgData = images.find(function(i) { return i.filename === imgFilename; });
                if (!imgData) continue;

                var manifestInfo = manifestData[imgFilename] || {};

                var spillRate = imgData.spill_rate ? imgData.spill_rate.toFixed(6) : '';
                var coverage = imgData.coverage ? imgData.coverage.toFixed(6) : '';
                var domClass = imgData.dominant_class || '';
                var caption = imgData.caption || '';

                rows.push([
                    imgFilename,
                    status,
                    manifestInfo.clean_path || '',
                    manifestInfo.clean_file_id || '',
                    manifestInfo.clean_camera || '',
                    manifestInfo.mask_file_id || '',
                    spillRate,
                    coverage,
                    domClass,
                    caption
                ].map(function(v) { return '"' + v + '"'; }).join(','));
            }

            var csvContent = [
                headers.join(','),
                rows.join(String.fromCharCode(10))
            ].join(String.fromCharCode(10));

            // 创建Blob并下载
            var blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
            var url = URL.createObjectURL(blob);
            var a = document.createElement('a');
            a.href = url;
            a.download = filename;
            a.click();
            URL.revokeObjectURL(url);

            alert('已导出 ' + rows.length + ' 条筛选结果' + String.fromCharCode(10) + '文件名: ' + filename);
        }

        function jumpToImage() {
            var num = parseInt(document.getElementById('jump-to').value);
            if (num > 0 && num <= filteredImages.length) {
                currentBatch = Math.floor((num - 1) / batchSize);
                renderGrid();
                setTimeout(function() { openModal(num - 1); }, 100);
            }
        }

        // 事件监听
        document.getElementById('status-filter').addEventListener('change', applyFilters);
        document.getElementById('sort-by').addEventListener('change', applyFilters);
        document.getElementById('spill-filter').addEventListener('input', function(e) {
            document.getElementById('spill-value').textContent = e.target.value + '%';
            applyFilters();
        });

        // 键盘快捷键
        document.addEventListener('keydown', function(e) {
            if (!document.getElementById('modal').classList.contains('active')) return;

            switch(e.key.toLowerCase()) {
                case 'a': acceptCurrent(); break;
                case 'r': rejectCurrent(); break;
                case 'c': toggleCompareMode(); break;
                case 'arrowleft': navigateImage(-1); break;
                case 'arrowright': navigateImage(1); break;
                case 'escape': closeModal(); break;
            }
        });

        // 初始化
        init();
    </script>
</body>
</html>
"""

# ============================================================================
# Flask应用
# ============================================================================

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# 全局变量
manifest_df = None
images_dir = None
data_dir = None


def load_manifest(data_dir_path):
    """加载manifest数据"""
    manifest_path = Path(data_dir_path) / "manifest.csv"
    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
        # 添加coverage列（如果不存在）
        if 'coverage' not in df.columns:
            df['coverage'] = df.get('mask_coverage', 0.5)
        return df
    return None


def get_image_list(data_dir_path):
    """获取图像列表"""
    img_dir = Path(data_dir_path) / "images"
    if not img_dir.exists():
        return []
    return sorted([f.name for f in img_dir.glob("*.png")])


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/images')
def api_images():
    """返回图像列表及元数据"""
    try:
        image_list = []
        for filename in get_image_list(data_dir):
            try:
                row = manifest_df[manifest_df['output_filename'] == filename]
                if len(row) > 0:
                    row = row.iloc[0]
                    # 构建干净帧的URL路径
                    clean_path = row.get('clean_path', '')
                    if clean_path:
                        # 从完整路径提取相对路径: camera_type/filename.jpg
                        try:
                            clean_path_parts = Path(clean_path).parts
                            if 'my_clean_frames' in clean_path_parts:
                                idx = clean_path_parts.index('my_clean_frames')
                                clean_rel_path = str(Path(*clean_path_parts[idx+1:]))
                            else:
                                clean_rel_path = ''
                        except Exception:
                            clean_rel_path = ''
                    else:
                        clean_rel_path = ''

                    image_list.append({
                        'filename': str(filename),
                        'spill_rate': float(row.get('spill_rate', 0.5)),
                        'coverage': float(row.get('coverage', 0.5)),
                        'dominant_class': int(row.get('mask_dominant_class', 3)),
                        'caption': str(row.get('caption', '')),
                        'clean_path': clean_rel_path,
                        'clean_file_id': str(row.get('clean_file_id', ''))
                    })
                else:
                    image_list.append({
                        'filename': str(filename),
                        'spill_rate': 0.5,
                        'coverage': 0.5,
                        'dominant_class': 3,
                        'caption': '',
                        'clean_path': '',
                        'clean_file_id': ''
                    })
            except Exception as e:
                app.logger.error(f"Error processing {filename}: {e}")
                # 添加默认条目
                image_list.append({
                    'filename': str(filename),
                    'spill_rate': 0.5,
                    'coverage': 0.5,
                    'dominant_class': 3,
                    'caption': '',
                    'clean_path': '',
                    'clean_file_id': ''
                })
        return jsonify(image_list)
    except Exception as e:
        app.logger.error(f"Error in api_images: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/images/<filename>')
def serve_image(filename):
    """提供生成的脏污图像文件"""
    img_path = Path(images_dir) / filename
    if img_path.exists():
        return send_file(img_path)
    return "Not found", 404


@app.route('/clean/<path:filepath>')
def serve_clean_image(filepath):
    """提供原始干净帧图像 (用于对比)"""
    # filepath格式: camera_type/filename.jpg
    clean_frames_base = "/home/yf/soiling_project/dataset/my_clean_frames"
    img_path = Path(clean_frames_base) / filepath
    if img_path.exists():
        return send_file(img_path)
    return "Not found", 404


@app.route('/api/decisions', methods=['GET', 'POST'])
def api_decisions():
    """获取或保存筛选决定"""
    decisions_file = Path(data_dir) / "filter_decisions.json"

    if request.method == 'GET':
        if decisions_file.exists():
            with open(decisions_file, 'r') as f:
                return jsonify(json.load(f))
        return jsonify({})

    if request.method == 'POST':
        decisions = request.json
        with open(decisions_file, 'w') as f:
            json.dump(decisions, f, indent=2)
        return jsonify({'status': 'ok'})


@app.route('/api/manifest')
def api_manifest():
    """返回manifest数据映射"""
    if manifest_df is None:
        return jsonify({})

    manifest_map = {}
    for _, row in manifest_df.iterrows():
        filename = row.get('output_filename', '')
        if filename:
            manifest_map[filename] = {
                'clean_path': str(row.get('clean_path', '')),
                'clean_file_id': str(row.get('clean_file_id', '')),
                'clean_camera': str(row.get('clean_camera', '')),
                'mask_file_id': str(row.get('mask_file_id', '')),
            }
    return jsonify(manifest_map)


@app.route('/api/export')
def export_results():
    """导出筛选结果"""
    decisions_file = Path(data_dir) / "filter_decisions.json"

    if not decisions_file.exists():
        return "No decisions found", 404

    with open(decisions_file, 'r') as f:
        decisions = json.load(f)

    # 创建结果DataFrame
    results = []
    for filename, status in decisions.items():
        row = manifest_df[manifest_df['output_filename'] == filename]
        if len(row) > 0:
            result = row.iloc[0].to_dict()
            result['filter_status'] = status
            results.append(result)

    df = pd.DataFrame(results)

    # 生成CSV
    output = Path(data_dir) / f"filtered_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(output, index=False)

    return send_file(output, as_attachment=True, download_name=output.name)


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="合成图像筛选工具")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="批量生成输出目录 (包含images/和manifest.csv)")
    parser.add_argument("--port", type=int, default=5000,
                       help="Web服务端口")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                       help="Web服务主机")

    args = parser.parse_args()

    global manifest_df, images_dir, data_dir
    data_dir = Path(args.data_dir)
    images_dir = data_dir / "images"

    # 加载manifest
    manifest_df = load_manifest(data_dir)
    if manifest_df is None:
        print(f"⚠️ 警告: 未找到manifest.csv")

    # 检查图像目录
    if not images_dir.exists():
        print(f"❌ 错误: 图像目录不存在: {images_dir}")
        return

    image_count = len(get_image_list(data_dir))
    print(f"📁 数据目录: {data_dir}")
    print(f"🖼️ 图像数量: {image_count}")
    print(f"🚀 启动Web服务: http://{args.host}:{args.port}")
    print("\n按 Ctrl+C 停止服务")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
