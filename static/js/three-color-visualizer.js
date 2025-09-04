/**
 * Three.js Color Visualizer for ColorAnalyzer - Next Gen
 * Immersive 3D color space visualization with advanced effects and controls.
 */

class ColorVisualizer3D {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) throw new Error(`Container '${containerId}' not found`);

        this.isInitialized = false;
        this.colorData = [];
        this.hsvData = [];
        this.particles = null;
        this.velocities = null;
        this.currentMode = 'rgb';
        this.controlsMode = 'orbit';

        this.clock = new THREE.Clock();

        this.init();
    }

    init() {
        // Scene
        this.scene = new THREE.Scene();

        // Camera
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 5000);
        this.camera.position.set(0, 150, 400);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.toneMapping = THREE.ReinhardToneMapping;
        this.container.appendChild(this.renderer.domElement);

        // Controls
        this.setupControls();

        // Skybox
        this.createSkybox();

        // Post-processing (Bloom Effect)
        this.setupPostProcessing();

        // Event Listeners
        window.addEventListener('resize', () => this.onWindowResize());

        this.isInitialized = true;
        this.animate();
    }

    setupControls() {
        // Orbit Controls (Default)
        this.orbitControls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.orbitControls.enableDamping = true;
        this.orbitControls.dampingFactor = 0.05;
        this.orbitControls.minDistance = 50;
        this.orbitControls.maxDistance = 1000;
        this.orbitControls.target.set(0, 50, 0);

        // Pointer Lock Controls (Fly)
        this.pointerLockControls = new THREE.PointerLockControls(this.camera, this.renderer.domElement);
        const instructions = document.getElementById('pointer-lock-instructions');
        this.pointerLockControls.addEventListener('lock', () => instructions.style.display = 'none');
        this.pointerLockControls.addEventListener('unlock', () => instructions.style.display = 'block');
        this.container.addEventListener('click', () => {
            if (this.controlsMode === 'fly') this.pointerLockControls.lock();
        });

        this.orbitControls.enabled = true;
        this.pointerLockControls.enabled = false;
    }

    createSkybox() {
        const loader = new THREE.CubeTextureLoader();
        loader.setPath('static/skybox/'); // Assumes skybox images are in this folder
        const textureCube = loader.load([
            'px.png', 'nx.png',
            'py.png', 'ny.png',
            'pz.png', 'nz.png'
        ]);
        this.scene.background = textureCube;
    }

    setupPostProcessing() {
        const renderScene = new THREE.RenderPass(this.scene, this.camera);

        this.bloomPass = new THREE.UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 1.5, 0.4, 0.85);
        this.bloomPass.threshold = 0;
        this.bloomPass.strength = 1.5; // Default strength
        this.bloomPass.radius = 0;

        this.composer = new THREE.EffectComposer(this.renderer);
        this.composer.addPass(renderScene);
        this.composer.addPass(this.bloomPass);
    }

    loadColorData(colorData, hsvData) {
        this.colorData = colorData;
        this.hsvData = hsvData;
        this.createParticles();
    }

    createParticles() {
        if (this.particles) {
            this.scene.remove(this.particles);
            this.particles.geometry.dispose();
            this.particles.material.dispose();
        }

        const data = this.currentMode === 'hsv' ? this.hsvData : this.colorData;
        if (!data || data.length === 0) return;

        const geometry = new THREE.BufferGeometry();
        const positions = [];
        const colors = [];
        this.velocities = [];

        const maxPoints = 20000;
        const step = Math.max(1, Math.floor(data.length / maxPoints));

        for (let i = 0; i < data.length; i += step) {
            const p = data[i];
            if (this.currentMode === 'rgb') {
                positions.push(p.r - 128, p.g - 128, p.b - 128);
            } else { // hsv
                positions.push(p.x, p.y - 50, p.z);
            }
            colors.push(p.r / 255, p.g / 255, p.b / 255);
            this.velocities.push((Math.random() - 0.5) * 0.5, (Math.random() - 0.5) * 0.5, (Math.random() - 0.5) * 0.5);
        }

        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
            size: 2.5,
            vertexColors: true,
            blending: THREE.AdditiveBlending,
            transparent: true,
            opacity: 0.9
        });

        this.particles = new THREE.Points(geometry, material);
        this.scene.add(this.particles);
    }

    animateParticles() {
        if (!this.particles || !this.velocities) return;

        const positions = this.particles.geometry.attributes.position.array;
        const numParticles = positions.length / 3;

        for (let i = 0; i < numParticles; i++) {
            const i3 = i * 3;
            positions[i3] += this.velocities[i3];
            positions[i3 + 1] += this.velocities[i3 + 1];
            positions[i3 + 2] += this.velocities[i3 + 2];

            // Simple bounds check to keep particles contained
            if (positions[i3] > 200 || positions[i3] < -200) this.velocities[i3] *= -1;
            if (positions[i3 + 1] > 200 || positions[i3 + 1] < -200) this.velocities[i3 + 1] *= -1;
            if (positions[i3 + 2] > 200 || positions[i3 + 2] < -200) this.velocities[i3 + 2] *= -1;
        }
        this.particles.geometry.attributes.position.needsUpdate = true;
    }

    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.composer.setSize(window.innerWidth, window.innerHeight);
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        const delta = this.clock.getDelta();

        if (this.orbitControls.enabled) {
            this.orbitControls.update();
        }
        if (this.pointerLockControls.isLocked === true) {
            // Add movement logic for fly controls here if desired
        }

        // this.animateParticles();
        this.composer.render(delta);
    }

    // --- Public API ---

    setVisualizationMode(mode) {
        if (this.currentMode === mode) return;
        this.currentMode = mode;
        this.createParticles(); // Re-create particles with the new data source
    }

    setControls(mode) {
        if (this.controlsMode === mode) return;
        this.controlsMode = mode;

        const instructions = document.getElementById('pointer-lock-instructions');
        const container = document.getElementById('three-container');

        if (mode === 'fly') {
            this.orbitControls.enabled = false;
            this.pointerLockControls.enabled = true;
            instructions.style.display = 'flex';
            container.classList.add('fly-mode');
        } else { // orbit
            this.pointerLockControls.unlock();
            this.orbitControls.enabled = true;
            this.pointerLockControls.enabled = false;
            instructions.style.display = 'none';
            container.classList.remove('fly-mode');
        }
    }

    setBloomStrength(strength) {
        if (this.bloomPass) {
            this.bloomPass.strength = strength;
        }
    }
}

// Export for use in other scripts
window.ColorVisualizer3D = ColorVisualizer3D;
