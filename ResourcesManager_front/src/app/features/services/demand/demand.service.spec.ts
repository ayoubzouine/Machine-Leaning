import { TestBed } from '@angular/core/testing';

import { DemandsService } from './demand.service';

describe('DemandsService', () => {
  let service: DemandsService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(DemandsService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
