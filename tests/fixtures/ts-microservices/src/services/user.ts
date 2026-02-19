import { BaseService, db, publish, EventType } from '../shared';
import type { User } from '../types';

export class UserService extends BaseService {
  constructor() {
    super('user');
  }

  createUser(email: string): User {
    this.log('createUser');
    db.query('insert into users', [email]);
    publish({ type: EventType.UserCreated, payload: { email } });
    return { id: 'u1', email };
  }
}

export function loadUser(id: string): User {
  db.query('select * from users', [id]);
  return { id, email: 'demo@example.com' };
}
