export interface TagBody {
  x: number;
  y: number;
  vx: number;
  vy: number;
  width: number;
  height: number;
  held: boolean;
}

export function stepTags(bodies: TagBody[], width: number, height: number, dt: number) {
  for (const body of bodies) {
    if (body.held) continue;
    body.vy += 850 * dt;
    body.vx *= Math.exp(-1.8 * dt);
    body.x += body.vx * dt;
    body.y += body.vy * dt;
  }

  for (let pass = 0; pass < 5; pass++) {
    for (let i = 0; i < bodies.length; i++) {
      for (let j = i + 1; j < bodies.length; j++) {
        const a = bodies[i];
        const b = bodies[j];
        const dx = a.x + a.width / 2 - b.x - b.width / 2;
        const dy = a.y + a.height / 2 - b.y - b.height / 2;
        const overlapX = (a.width + b.width) / 2 + 8 - Math.abs(dx);
        const overlapY = (a.height + b.height) / 2 + 8 - Math.abs(dy);
        if (overlapX <= 0 || overlapY <= 0 || (a.held && b.held)) continue;
        const axis = overlapX < overlapY ? 'x' : 'y';
        const velocity = axis === 'x' ? 'vx' : 'vy';
        const sign = (axis === 'x' ? dx : dy) >= 0 ? 1 : -1;
        const overlap = axis === 'x' ? overlapX : overlapY;
        const invA = a.held ? 0 : 1;
        const invB = b.held ? 0 : 1;
        a[axis] += sign * overlap * invA / (invA + invB);
        b[axis] -= sign * overlap * invB / (invA + invB);
        const relativeVelocity = (a[velocity] - b[velocity]) * sign;
        if (relativeVelocity < 0) {
          const impulse = -1.25 * relativeVelocity / (invA + invB);
          a[velocity] += impulse * sign * invA;
          b[velocity] -= impulse * sign * invB;
        }
      }
    }
    for (const body of bodies) {
      if (body.held) continue;
      const maxX = Math.max(0, width - body.width);
      const maxY = Math.max(0, height - body.height);
      if (body.x < 0 || body.x > maxX) {
        body.x = Math.max(0, Math.min(maxX, body.x));
        body.vx *= -.35;
      }
      if (body.y < 0) {
        body.y = 0;
        body.vy = Math.abs(body.vy) * .3;
      }
      if (body.y > maxY) {
        body.y = maxY;
        body.vy = Math.abs(body.vy) < 35 ? 0 : -body.vy * .25;
        body.vx *= .85;
      }
    }
  }
}
