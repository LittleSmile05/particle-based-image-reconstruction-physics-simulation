import pygame
import sys
import math
import random
from PIL import Image
import numpy as np
import cv2  

# --- Constants ---
WIDTH, HEIGHT = 700, 700
GRAVITY = pygame.math.Vector2(0, 981)  # Pixels per second^2
BACKGROUND_COLOR = (0, 0, 0)
BOUNCE_LOSS = 0.9  # Energy retained after bouncing off walls
COLLISION_DAMPING = 0.95  # Energy retained after inter-ball collision
MIN_RADIUS = 2  # Minimum radius
MAX_RADIUS = 8  # Maximum radius (reduced for better detail)
SPAWN_STEP_INTERVAL = 1  # Spawn new ball every N steps
FIXED_DT = 1/60.0
MODE = "SPAWN"
TOTAL_STEPS = 400
SIMULATION_SUBSTEPS = 8  # Increase for more accuracy, decrease for performance

# --- Adaptive particle count parameters ---
MIN_PARTICLES = 1000  # Increased minimum number of particles
MAX_PARTICLES = 5000  # Increased maximum number of particles
COMPLEXITY_THRESHOLD_LOW = 0.1  # Below this complexity, use MIN_PARTICLES
COMPLEXITY_THRESHOLD_HIGH = 0.5  # Above this complexity, use MAX_PARTICLES

# --- Enhanced image representation parameters ---
SPRING_STRENGTH = 1.2  # Increased spring strength (was 0.4)
ATTRACTION_PHASE = 80  # Reduced wait time before pulling balls to positions
DAMPING = 0.95  # Increased damping to help balls settle faster

# --- Ball Class ---
class Ball:
    """Represents a single ball in the physics simulation."""
    def __init__(self, position, radius, step_added, color=None):
        self.position = pygame.math.Vector2(position)
        # Verlet integration uses current and previous position to infer velocity
        self.old_position = pygame.math.Vector2(position)
        self.acceleration = pygame.math.Vector2(0, 0)
        self.radius = radius
        self.mass = math.pi * radius**2 
        self.step_added = step_added
        self.color = color if color else (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        self.original_position = pygame.math.Vector2(position)
        self.target_position = pygame.math.Vector2(position)

    def update(self, dt):
        """Updates the ball's position using Verlet integration."""
        # Calculate velocity based on position change
        velocity = self.position - self.old_position
        # Apply damping to gradually reduce overall energy
        velocity *= DAMPING
        self.old_position = pygame.math.Vector2(self.position)
        # Perform Verlet integration: pos += vel + acc * dt^2
        self.position += velocity + self.acceleration * dt * dt
        # Reset acceleration for the next frame/substep
        self.acceleration = pygame.math.Vector2(0, 0)

    def accelerate(self, force):
        """Applies a force to the ball (F = ma -> a = F/m)."""
        # Ensure mass is not zero to avoid division by zero
        if self.mass > 0:
            self.acceleration += force / self.mass

    def apply_spring_force(self, current_step):
        """Apply a spring force to pull the ball back toward its original position."""
        if current_step < ATTRACTION_PHASE:
            return  # Don't apply spring force until after attraction phase
            
        # Calculate vector from current to target position
        to_target = self.target_position - self.position
        distance = to_target.length()
        spring_strength_factor = min(3.0, 1.0 + (distance / 100.0))
        time_factor = min(1.0, (current_step - ATTRACTION_PHASE) / 100.0)
        effective_spring_strength = SPRING_STRENGTH * spring_strength_factor * time_factor
        
        spring_force = to_target * effective_spring_strength
        
        self.accelerate(spring_force)

    def apply_constraints(self):
        """Keeps the ball within the screen boundaries."""
        velocity = self.position - self.old_position

        # Left boundary
        if self.position.x - self.radius < 0:
            self.position.x = self.radius
            self.old_position.x = self.position.x + (self.position.x - self.old_position.x) * -BOUNCE_LOSS
        # Right boundary
        elif self.position.x + self.radius > WIDTH:
            self.position.x = WIDTH - self.radius
            self.old_position.x = self.position.x + (self.position.x - self.old_position.x) * -BOUNCE_LOSS
        # Top boundary
        if self.position.y - self.radius < 0:
            self.position.y = self.radius
            self.old_position.y = self.position.y + (self.position.y - self.old_position.y) * -BOUNCE_LOSS
        # Bottom boundary
        elif self.position.y + self.radius > HEIGHT:
            self.position.y = HEIGHT - self.radius
            self.old_position.y = self.position.y + (self.position.y - self.old_position.y) * -BOUNCE_LOSS

    def draw(self, screen):
        """Draws the ball on the Pygame screen."""
        draw_pos = (int(self.position.x), int(self.position.y))
        pygame.draw.circle(screen, self.color, draw_pos, int(self.radius))

# --- Simulation Class ---
class Simulation:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Improved Image-based Physics Simulation")
        self.balls = []
        self.font = pygame.font.Font(None, 30)
        self.current_step = 0
        self.clock = pygame.time.Clock()
        random.seed(42)

        self.input_image = None
        self.image_pixels = None
        self.sampled_points = []
        self.max_objects = MIN_PARTICLES  
        
        try:
            self.input_image = Image.open('input_imagee.jpg').convert('RGB')
            self.input_image = self.input_image.resize((WIDTH, HEIGHT))
            self.image_pixels = np.array(self.input_image)
            
            self.calculate_particle_count()
            self.sample_points_from_image()
            
            print(f"Image complexity assessment: Using {self.max_objects} particles")
        except Exception as e:
            print(f"Error loading image: {e}")
            print("No input image found - will use default particle count")
            self.max_objects = MIN_PARTICLES

    def calculate_particle_count(self):
        """Calculate appropriate number of particles based on image complexity."""
        if self.image_pixels is None:
            self.max_objects = MIN_PARTICLES
            return
            
        # Convert to grayscale for analysis
        gray = np.mean(self.image_pixels, axis=2).astype(np.uint8)
        # 1. Edge density (using Canny edge detection)
        try:
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / (WIDTH * HEIGHT)
        except:
            # Fallback if OpenCV not available
            grad_x = np.abs(np.gradient(gray, axis=1))
            grad_y = np.abs(np.gradient(gray, axis=0))
            edge_strength = np.sqrt(grad_x**2 + grad_y**2)
            edge_density = np.sum(edge_strength > 15) / (WIDTH * HEIGHT)
        
        # 2. Color variance
        r_var = np.var(self.image_pixels[:,:,0]) / 255**2
        g_var = np.var(self.image_pixels[:,:,1]) / 255**2
        b_var = np.var(self.image_pixels[:,:,2]) / 255**2
        color_complexity = (r_var + g_var + b_var) / 3
        
        # 3. Entropy (measure of randomness/information)
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist = hist / np.sum(hist)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        normalized_entropy = entropy / 8.0  # Maximum entropy for 8-bit image
        
        # Combine measures to get overall complexity score
        complexity_score = 0.5 * edge_density + 0.3 * color_complexity + 0.2 * normalized_entropy
        
        # Map complexity score to particle count using a smooth function
        if complexity_score <= COMPLEXITY_THRESHOLD_LOW:
            self.max_objects = MIN_PARTICLES
        elif complexity_score >= COMPLEXITY_THRESHOLD_HIGH:
            self.max_objects = MAX_PARTICLES
        else:
            # Linear interpolation between thresholds
            normalized_score = (complexity_score - COMPLEXITY_THRESHOLD_LOW) / (COMPLEXITY_THRESHOLD_HIGH - COMPLEXITY_THRESHOLD_LOW)
            self.max_objects = int(MIN_PARTICLES + normalized_score * (MAX_PARTICLES - MIN_PARTICLES))
        
        # Round to nearest hundred for cleaner numbers
        self.max_objects = int(round(self.max_objects / 100) * 100)
        
        # Log detailed complexity analysis
        print(f"Image complexity analysis:")
        print(f"- Edge density: {edge_density:.4f}")
        print(f"- Color variance: {color_complexity:.4f}")
        print(f"- Entropy: {normalized_entropy:.4f}")
        print(f"- Overall complexity score: {complexity_score:.4f}")

    def sample_points_from_image(self):
        """Sample points from the image based on visual importance using improved algorithms."""
        if self.image_pixels is None:
            return
        
        # Step 1: Convert to grayscale for analysis
        gray = np.mean(self.image_pixels, axis=2).astype(np.uint8)
        
        # Step 2: Enhanced edge detection with multi-scale approach
        try:
            # Use multiple edge detection thresholds to capture different levels of detail
            edges1 = cv2.Canny(gray, 30, 100)  # More sensitive, catches subtle edges
            edges2 = cv2.Canny(gray, 100, 200)  # Medium sensitivity
            edges3 = cv2.Canny(gray, 200, 250)  # Less sensitive, only strong edges
            
            # Combine edge maps with different weights
            edge_strength = (edges1 * 0.2 + edges2 * 0.5 + edges3 * 0.3) / 255.0
            
            # Dilate edges for better coverage of important areas
            kernel = np.ones((2,2), np.uint8)
            edge_strength = cv2.dilate(edge_strength, kernel, iterations=1)
        except:
            # Fallback gradient-based edge detection with smoothing
            grad_x = np.abs(np.gradient(gray, axis=1))
            grad_y = np.abs(np.gradient(gray, axis=0))
            edge_strength = np.sqrt(grad_x**2 + grad_y**2)
            # Normalize edge strength
            if np.max(edge_strength) > 0:
                edge_strength = edge_strength / np.max(edge_strength)
        
        # Step 3: Calculate importance metrics
        # Brightness - both very dark and very bright areas matter
        brightness = gray / 255.0
        brightness_importance = np.abs(brightness - 0.5) * 2
        
        # Color saturation - more saturated colors are visually important
        r, g, b = self.image_pixels[:,:,0], self.image_pixels[:,:,1], self.image_pixels[:,:,2]
        max_c = np.maximum(np.maximum(r, g), b) / 255.0
        min_c = np.minimum(np.minimum(r, g), b) / 255.0
        saturation = (max_c - min_c) / (max_c + 1e-10)
        
        # Color uniqueness - areas with uncommon colors should be preserved
        # Create a simplified color histogram
        colors = self.image_pixels.reshape(-1, 3)
        color_bins = colors // 32  # Divide colors into bins
        color_keys = color_bins[:, 0] * 64 + color_bins[:, 1] * 8 + color_bins[:, 2]
        unique_colors, counts = np.unique(color_keys, return_counts=True)
        color_frequency = dict(zip(unique_colors, counts / len(color_keys)))
        
        # Calculate color uniqueness map
        color_uniqueness = np.zeros_like(brightness)
        flat_keys = (self.image_pixels[:,:,0] // 32) * 64 + (self.image_pixels[:,:,1] // 32) * 8 + (self.image_pixels[:,:,2] // 32)
        for y in range(HEIGHT):
            for x in range(WIDTH):
                key = flat_keys[y, x]
                color_uniqueness[y, x] = 1.0 - min(1.0, color_frequency.get(key, 0) * 10)
        
        # Step 4: Combine factors with adaptive weights 
        # Check if the image has strong edges
        if np.mean(edge_strength) > 0.1:
            # Edge-dominant image (like line art, sketches)
            importance = (edge_strength * 0.7 + 
                         brightness_importance * 0.1 + 
                         saturation * 0.1 +
                         color_uniqueness * 0.1)
        else:
            # Color/tone dominant image
            importance = (edge_strength * 0.4 + 
                         brightness_importance * 0.2 + 
                         saturation * 0.2 +
                         color_uniqueness * 0.2)
        
        # Step 5: Add uniform background sampling to ensure coverage
        uniform_sampling = np.ones_like(importance) * 0.05
        importance = importance * 0.95 + uniform_sampling
        
        # Normalize to [0, 1]
        if np.max(importance) > 0:
            importance = importance / np.max(importance)
        
        # Create probability distribution based on importance
        prob_map = importance + 0.01  # Small baseline probability
        prob_map = prob_map / np.sum(prob_map)
        
        # Flatten for random sampling
        flat_prob = prob_map.flatten()
        indices = np.arange(flat_prob.size)
        
        # Sample points based on probability
        sampled_indices = np.random.choice(
            indices, 
            size=min(self.max_objects * 2, flat_prob.size),  # Sample more points than needed
            replace=False, 
            p=flat_prob
        )
        
        # Convert back to 2D coordinates and record color info
        temp_points = []
        for idx in sampled_indices:
            y, x = np.unravel_index(idx, importance.shape)
            r, g, b = self.image_pixels[y, x]
            # Store each point with its importance value
            point_importance = importance[y, x]
            temp_points.append((x, y, (int(r), int(g), int(b)), point_importance))
        
        # Sort by importance and take the most important points
        temp_points.sort(key=lambda p: p[3], reverse=True)
        self.sampled_points = [(x, y, color) for x, y, color, _ in temp_points[:self.max_objects]]
        
        # Ensure we have sufficient sampling for all important areas
        if len(self.sampled_points) < self.max_objects:
            # If we need more points, add some from a regular grid
            points_needed = self.max_objects - len(self.sampled_points)
            grid_size = int(math.sqrt(points_needed * 1.5))  # Oversample grid
            
            # Create grid points
            grid_points = []
            for i in range(grid_size):
                for j in range(grid_size):
                    x = int((i + 0.5) * WIDTH / grid_size)
                    y = int((j + 0.5) * HEIGHT / grid_size)
                    if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                        r, g, b = self.image_pixels[y, x]
                        grid_points.append((x, y, (int(r), int(g), int(b))))
            
            # Shuffle grid points and add what we need
            random.shuffle(grid_points)
            self.sampled_points.extend(grid_points[:points_needed])
        
        print(f"Sampled {len(self.sampled_points)} points from image")

    def add_ball(self, ball):
        """Adds a ball to the simulation, respecting the max object limit."""
        if len(self.balls) < self.max_objects:
            self.balls.append(ball)
        elif self.balls:  # If limit reached, remove the oldest ball
            self.balls.pop(0)
            self.balls.append(ball)

    def apply_gravity(self):
        """Applies gravity to all balls."""
        for ball in self.balls:
            ball.accelerate(GRAVITY)

    def apply_spring_forces(self):
        """Apply spring forces to pull balls toward their original positions."""
        for ball in self.balls:
            ball.apply_spring_force(self.current_step)

    def update_positions(self, dt):
        """Updates the position of each ball."""
        for ball in self.balls:
            ball.update(dt)

    def apply_constraints(self):
        """Applies screen boundary constraints to all balls."""
        for ball in self.balls:
            ball.apply_constraints()

    def solve_collisions(self):
        """Detects and resolves collisions between balls using spatial partitioning."""
        # Use spatial partitioning for more efficient collision detection
        cells = {}
        cell_size = max(MAX_RADIUS * 2, 20)  # Cell size based on max radius
        
        # Sort balls into grid cells
        for i, ball in enumerate(self.balls):
            # Calculate grid cell coordinates
            cell_x = int(ball.position.x / cell_size)
            cell_y = int(ball.position.y / cell_size)
            
            # Add ball to all relevant cells (could be in up to 4 cells if on boundary)
            for dx in range(2):
                for dy in range(2):
                    cx = cell_x + dx
                    cy = cell_y + dy
                    if (cx, cy) not in cells:
                        cells[(cx, cy)] = []
                    cells[(cx, cy)].append(i)
        
        # Check collisions within each cell
        checked_pairs = set()
        for cell_indices in cells.values():
            for i in range(len(cell_indices)):
                idx1 = cell_indices[i]
                ball_1 = self.balls[idx1]
                
                for j in range(i + 1, len(cell_indices)):
                    idx2 = cell_indices[j]
                    
                    # Skip if this pair has been checked already
                    if (idx1, idx2) in checked_pairs or (idx2, idx1) in checked_pairs:
                        continue
                    
                    checked_pairs.add((idx1, idx2))
                    ball_2 = self.balls[idx2]

                    # Vector from ball_1 center to ball_2 center
                    collision_axis = ball_1.position - ball_2.position
                    dist_sq = collision_axis.length_squared()
                    min_dist = ball_1.radius + ball_2.radius

                    # Check if balls are overlapping (using squared distance for efficiency)
                    if dist_sq < min_dist * min_dist and dist_sq > 0:  # Ensure dist_sq is not zero
                        dist = math.sqrt(dist_sq)
                        # Normalized collision axis
                        normal = collision_axis / dist
                        # Amount of overlap
                        overlap = (min_dist - dist) * 0.5  # Divide by 2 as both balls move

                        # Separate the balls based on mass (lighter moves more)
                        # Calculate total mass for mass ratio calculation
                        total_mass = ball_1.mass + ball_2.mass
                        if total_mass == 0:  # Avoid division by zero if both masses are somehow zero
                            mass_ratio_1 = 0.5
                            mass_ratio_2 = 0.5
                        else:
                            mass_ratio_1 = ball_2.mass / total_mass
                            mass_ratio_2 = ball_1.mass / total_mass

                        # Move balls apart along the normal vector
                        separation_vector = normal * overlap
                        ball_1.position += separation_vector * mass_ratio_1
                        ball_2.position -= separation_vector * mass_ratio_2

                        # --- Calculate velocity changes after collision ---
                        vel1 = ball_1.position - ball_1.old_position
                        vel2 = ball_2.position - ball_2.old_position
                        
                        # Project velocities onto collision normal
                        vel1_normal_mag = vel1.dot(normal)
                        vel2_normal_mag = vel2.dot(normal)
                        
                        # Only exchange momentum along normal if balls are moving toward each other
                        if vel1_normal_mag - vel2_normal_mag < 0:
                            vel1_normal = vel1_normal_mag * normal
                            vel2_normal = vel2_normal_mag * normal
                            
                            # Compute tangential components (preserved in collision)
                            vel1_tang = vel1 - vel1_normal
                            vel2_tang = vel2 - vel2_normal
                            
                            # Exchange normal components of velocity (with damping)
                            new_vel1 = vel1_tang + vel2_normal * COLLISION_DAMPING
                            new_vel2 = vel2_tang + vel1_normal * COLLISION_DAMPING
                            
                            # Update old positions to reflect new velocities
                            ball_1.old_position = ball_1.position - new_vel1
                            ball_2.old_position = ball_2.position - new_vel2

    def update(self, dt):
        """Performs a full simulation step, including sub-steps."""
        sub_dt = dt / SIMULATION_SUBSTEPS
        for _ in range(SIMULATION_SUBSTEPS):
            # In early phase, reduce gravity to let initial structure form
            if self.current_step < ATTRACTION_PHASE * 0.5:
                reduced_gravity = GRAVITY * (self.current_step / (ATTRACTION_PHASE * 0.5))
                for ball in self.balls:
                    ball.accelerate(reduced_gravity)
            else:
                self.apply_gravity()
                
            # Apply spring forces
            self.apply_spring_forces()
            self.update_positions(sub_dt)
            self.apply_constraints()
            self.solve_collisions()

    def draw(self):
        """Draws all elements of the simulation."""
        self.screen.fill(BACKGROUND_COLOR)
        
        # Optional: Draw original image as background at reduced opacity
        # Uncomment for debugging purposes
        '''
        if self.input_image:
            pygame_image = pygame.image.fromstring(
                self.input_image.tobytes(), self.input_image.size, self.input_image.mode)
            # Create a transparent surface
            transparent = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            transparent.fill((0, 0, 0, 128))  # Semi-transparent black
            # Blit the original image
            self.screen.blit(pygame_image, (0, 0))
            # Blit the transparent layer over it
            self.screen.blit(transparent, (0, 0))
        '''
        
        # Draw all balls
        for ball in self.balls:
            ball.draw(self.screen)

        # Display step and object count
        stats_text = self.font.render(f"Step: {self.current_step}/{TOTAL_STEPS} | Objects: {len(self.balls)}/{self.max_objects}", True, (200, 200, 200))
        self.screen.blit(stats_text, (10, 10))

        # Add phase indicator
        if self.current_step < ATTRACTION_PHASE:
            phase_text = self.font.render("Initial Phase: Building Structure", True, (200, 200, 100))
        else:
            phase_text = self.font.render("Spring Phase: Forming Image", True, (100, 200, 100))
        self.screen.blit(phase_text, (10, 40))

        pygame.display.flip()

    def save_ball_data(self):
        """Saves the final ball positions and colors to a CSV file."""
        with open('ball_spawns.csv', 'w') as f:
            for ball in self.balls:
                f.write(f"{ball.step_added},{ball.position.x},{ball.position.y},{ball.radius},{ball.old_position.x},{ball.old_position.y},{ball.color[0]},{ball.color[1]},{ball.color[2]}\n")

    def run(self):
        """Main simulation loop using fixed timesteps."""
        running = True
        balls_spawned = 0

        if MODE == "SPAWN":
            # Clear the CSV file
            with open('ball_spawns.csv', 'w') as f:
                f.write("")
                
            # Prepare to spawn balls from sampled image points
            if self.sampled_points:
                # We want a mix of randomness and structure
                # Add random jitter to initial positions but maintain distribution
                jittered_points = []
                for x, y, color in self.sampled_points:
                    # Add small random offset (max 5 pixels) to prevent perfect grid alignment
                    jitter_x = x + random.uniform(-2, 2)
                    jitter_y = y + random.uniform(-2, 2)
                    # Keep within bounds
                    jitter_x = max(0, min(WIDTH-1, jitter_x))
                    jitter_y = max(0, min(HEIGHT-1, jitter_y))
                    jittered_points.append((jitter_x, jitter_y, color))
                self.sampled_points = jittered_points
        else:
            # Read all balls from CSV at start
            try:
                with open('ball_spawns.csv', 'r') as f:
                    for line in f:
                        step, x, y, radius, old_x, old_y, r, g, b = map(float, line.strip().split(','))
                        if balls_spawned >= self.max_objects:
                            break
                        new_ball = Ball((x, y), radius, int(step), (int(r), int(g), int(b)))
                        new_ball.old_position = pygame.math.Vector2(old_x, old_y)
                        new_ball.original_position = pygame.math.Vector2(x, y)
                        new_ball.target_position = pygame.math.Vector2(x, y)
                        self.add_ball(new_ball)
                        balls_spawned += 1
            except Exception as e:
                print(f"Error reading ball data: {e}")

        # Dynamic adjustment of total steps based on particle count
        adjusted_steps = min(TOTAL_STEPS, max(300, int(self.max_objects * 0.15)))  # Reduce factor for faster convergence
        print(f"Adaptive simulation will run for {adjusted_steps} steps")

        while running and self.current_step < adjusted_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    # Add keyboard shortcuts
                    if event.key == pygame.K_s:  # Save current state
                        self.save_ball_data()
                        print("Saved ball data")
                    elif event.key == pygame.K_r:  # Reset balls to target positions
                        for ball in self.balls:
                            ball.position = pygame.math.Vector2(ball.target_position)
                            ball.old_position = pygame.math.Vector2(ball.target_position)
                        print("Reset balls to target positions")
                    
            self.clock.tick(60)
            
            # Calculate spawn rate to distribute particles evenly over the simulation
            spawn_rate = max(1, min(10, int(self.max_objects / (adjusted_steps * 0.5))))
            
            # Spawn balls at calculated rate
            if MODE == "SPAWN":
                for _ in range(spawn_rate):
                    if balls_spawned < min(self.max_objects, len(self.sampled_points)):
                        # Get the next point from our sampled image points
                        x, y, color = self.sampled_points[balls_spawned]
                        
                        # Adaptive radius based on particle count
                        if self.max_objects > 2000:
                            radius = random.uniform(MIN_RADIUS, MIN_RADIUS * 1.5)  # Smaller balls for high detail
                        else:
                            radius = random.uniform(MIN_RADIUS, MAX_RADIUS)
                            
                        new_ball = Ball((x, y), radius, self.current_step, color)
                        
                        # Set target position to original image point
                        new_ball.target_position = pygame.math.Vector2(x, y)
                        
                        # Minimal initial velocity for more controlled start
                        angle = random.uniform(0, 2 * math.pi)
                        blast_speed = random.uniform(2, 8)  # Much lower initial velocity
                        velocity = pygame.math.Vector2(
                            math.cos(angle) * blast_speed,
                            math.sin(angle) * blast_speed
                        )
                        new_ball.old_position = new_ball.position - velocity * FIXED_DT
                        
                        self.add_ball(new_ball)
                        balls_spawned += 1

            # Update simulation with fixed timestep
            self.update(FIXED_DT)
            self.draw()
            
            self.current_step += 1

        # Save ball data when simulation completes
        if MODE == "SPAWN":
            self.save_ball_data()
        
        # Keep window open until user exits
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:  # Save current state
                        self.save_ball_data()
                        print("Saved ball data")
                    elif event.key == pygame.K_r:  # Reset balls to target positions
                        for ball in self.balls:
                            ball.position = pygame.math.Vector2(ball.target_position)
                            ball.old_position = pygame.math.Vector2(ball.target_position)
                        print("Reset balls to target positions")
                    elif event.key == pygame.K_SPACE:  # Toggle pause
                        is_paused = not is_paused
                        print("Paused" if is_paused else "Resumed")
                    elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:  # Quit
                        running = False
            pygame.display.flip()
            self.clock.tick(30)
            
        pygame.quit()
        sys.exit()

    def toggle_background_image(self):
        """Toggle showing the original image in the background (for debugging)"""
        self.show_background = not self.show_background

# --- Optional: Add utility for image preprocessing ---
def preprocess_image(image_path, output_path=None, enhance_edges=True, enhance_colors=True):
    """
    Preprocess the input image to enhance edges and improve sampling quality
    Returns the processed image or saves to output_path if specified
    """
    try:
        # Open and convert to RGB
        img = Image.open(image_path).convert('RGB')
        
        # Convert to numpy array for processing
        img_array = np.array(img)
        
        # Resize if needed
        if img.width != WIDTH or img.height != HEIGHT:
            img = img.resize((WIDTH, HEIGHT), Image.LANCZOS)
            img_array = np.array(img)
        
        if enhance_edges:
            try:
                # Convert to grayscale for edge detection
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                
                # Use bilateral filter to smooth while preserving edges
                smooth = cv2.bilateralFilter(img_array, 9, 75, 75)
                
                # Detect edges
                edges = cv2.Canny(gray, 50, 150)
                
                # Dilate edges to make them more prominent
                kernel = np.ones((2, 2), np.uint8)
                edges = cv2.dilate(edges, kernel, iterations=1)
                
                # Create edge overlay
                edge_overlay = np.zeros_like(img_array)
                edge_overlay[edges > 0] = [0, 0, 0]  # Black edges
                
                # Blend original with edge overlay
                img_array = cv2.addWeighted(smooth, 0.8, edge_overlay, 0.2, 0)
            except:
                print("OpenCV edge enhancement failed, using basic processing")
                # Basic edge enhancement if OpenCV methods fail
                img_array = img_array.astype(np.float32)
                blurred = img_array.copy()
                for i in range(3):  # Simple blur
                    blurred[1:-1, 1:-1, i] = 0.25 * img_array[1:-1, 1:-1, i] + \
                                            0.125 * img_array[:-2, 1:-1, i] + \
                                            0.125 * img_array[2:, 1:-1, i] + \
                                            0.125 * img_array[1:-1, :-2, i] + \
                                            0.125 * img_array[1:-1, 2:, i]
                # Enhance edges by subtracting blurred image
                img_array = img_array + 0.7 * (img_array - blurred)
                img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        if enhance_colors:
            try:
                hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
                
                hsv[:, :, 1] = hsv[:, :, 1] * 1.2
                hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
                
                img_array = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            except:
                print("Color enhancement failed, using basic method")
           
                img_array = img_array.astype(np.float32)
                gray_avg = np.mean(img_array, axis=2, keepdims=True)
                img_array = gray_avg + 1.2 * (img_array - gray_avg)
                img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        processed_img = Image.fromarray(img_array)
        
        if output_path:
            processed_img.save(output_path)
            print(f"Preprocessed image saved to {output_path}")
            
        return processed_img
    
    except Exception as e:
        print(f"Image preprocessing failed: {e}")
        return None

def parse_arguments():
    """Parse command line arguments"""
    import argparse
    parser = argparse.ArgumentParser(description='Image-based Physics Simulation')
    parser.add_argument('--image', type=str, default='input_image.jpg', 
                        help='Path to input image')
    parser.add_argument('--csv', type=str, default='ball_spawns.csv',
                        help='Path to CSV file for ball data')  # Added this line
    parser.add_argument('--mode', type=str, default='SPAWN', choices=['SPAWN', 'LOAD'],
                        help='Simulation mode: SPAWN new balls or LOAD from CSV')
    parser.add_argument('--preprocess', action='store_true',
                        help='Preprocess the input image before simulation')
    parser.add_argument('--steps', type=int, default=TOTAL_STEPS,
                        help='Number of simulation steps')
    parser.add_argument('--min-particles', type=int, default=MIN_PARTICLES,
                        help='Minimum number of particles')
    parser.add_argument('--max-particles', type=int, default=MAX_PARTICLES,
                        help='Maximum number of particles')
    parser.add_argument('--spring-strength', type=float, default=SPRING_STRENGTH,
                        help='Spring strength for image formation')
    parser.add_argument('--gravity', type=float, default=GRAVITY[1],
                        help='Gravity strength')
    return parser.parse_args()

if __name__ == "__main__":
    try:
        args = parse_arguments()
        
        mode = args.mode
        total_steps = args.steps
        min_particles = args.min_particles
        max_particles = args.max_particles
        spring_strength = args.spring_strength
        gravity = pygame.math.Vector2(0, args.gravity)
        
        if args.preprocess:
            input_path = args.image
            processed_path = 'processed_' + input_path
            preprocess_image(input_path, processed_path, enhance_edges=True, enhance_colors=True)
            image_path = processed_path
        else:
            image_path = args.image
            
        simulation = Simulation(
            mode=mode,
            total_steps=total_steps,
            min_particles=min_particles,
            max_particles=max_particles,
            spring_strength=spring_strength,
            gravity=gravity,
            image_path=image_path
        )
        simulation.run()
    except Exception as e:
        print(f"Error processing command line arguments: {e}")
        print("Using defaults")
        simulation = Simulation()
        simulation.run()